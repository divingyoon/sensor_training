import argparse
import json
import os
import struct
import sys
import threading
import time
from ctypes import *
from datetime import datetime
from queue import Empty, Queue

try:
    import serial
except ImportError:
    serial = None

try:
    import can
except ImportError:
    can = None

try:
    from loadcell_bin_logger import (
        DEFAULT_BAUD as LOADCELL_BAUD,
        DEFAULT_PORT as LOADCELL_PORT,
        DEFAULT_READ_SIZE as LOADCELL_READ_SIZE,
        DEFAULT_RX_BUFFER_SIZE as LOADCELL_RX_BUFFER_SIZE,
        DEFAULT_TIMEOUT as LOADCELL_TIMEOUT,
        MAGIC as LOADCELL_MAGIC,
        RECORD_STRUCT as LOADCELL_RECORD_STRUCT,
        open_serial as open_loadcell_serial,
    )
except ImportError:
    LOADCELL_BAUD = 115200
    LOADCELL_PORT = "COM9"
    LOADCELL_READ_SIZE = 8192
    LOADCELL_RX_BUFFER_SIZE = 1024 * 1024
    LOADCELL_TIMEOUT = 0.01
    LOADCELL_MAGIC = "LOADCELL_BIN_V1"
    LOADCELL_RECORD_STRUCT = struct.Struct("<QI")
    open_loadcell_serial = None


DLL_PATH = r"C:\Program Files (x86)\PAIX\NMC\DLL\x64\NMC2.dll"
DUE_PORT = "COM11"
DUE_BAUD_RATE = 250000

CAN_INTERFACE = "ixxat"
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
AFD50_FORCE_ID = 0x01A

DEV_NO = 11
GROUP_NO = 0
TARGET_HZ = 200
POLLING_INTERVAL = 1.0 / TARGET_HZ
DURATION_LIMIT = 4 * 60 * 60
READY_TIMEOUT = 15.0
START_TRIGGER_TIMEOUT = 0.0
ETHERMOTION_IDLE_TIMEOUT = 10.0
DUE_PRESTART_PROBE_TIMEOUT = 2.0
AFD50_PRESTART_PROBE_TIMEOUT = 2.0
LOADCELL_PRESTART_PROBE_TIMEOUT = 2.0

NUM_SENSORS = 16
FIFO_FRAMES = 10
DUE_FRAME_HEADER = 0xAA
DUE_FRAME_FOOTER = 0x55
DUE_PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4
DUE_PACKET_FORMAT = "<" + ("I" * NUM_SENSORS * FIFO_FRAMES)

# Binary records are intentionally raw. Physical conversion is done offline.
DUE_RECORD_STRUCT = struct.Struct("<Q")  # elapsed_ns + raw 640-byte Arduino payload
AFD50_RECORD_STRUCT = struct.Struct("<QH8s")  # elapsed_ns, arbitration_id, raw CAN data
ETHERMOTION_AXES = [0, 1, 2, 3]
ETHERMOTION_RECORD_STRUCT = struct.Struct("<Qdddd")  # elapsed_ns, x_cmd, y_cmd, z_cmd, u_cmd
BUFFER_RECORDS = 8192

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

dirs = {
    "due": os.path.join(base_dir, "due data"),
    "afd": os.path.join(base_dir, "afd_50 data"),
    "stage": os.path.join(base_dir, "ethermotion data"),
    "loadcell": os.path.join(base_dir, "loadcell data"),
}
for directory in dirs.values():
    os.makedirs(directory, exist_ok=True)

due_queue = Queue()
afd_queue = Queue()
loadcell_queue = Queue()
is_running = True
log_start_ns = 0
logging_started = threading.Event()
due_ready = threading.Event()
afd_ready = threading.Event()
loadcell_ready = threading.Event()
reader_errors = Queue()


def elapsed_ns():
    return time.perf_counter_ns() - log_start_ns


def payload_to_rows(payload):
    values = struct.unpack(DUE_PACKET_FORMAT, payload)
    return [
        [
            values[sensor_i * FIFO_FRAMES + frame_j]
            for sensor_i in range(NUM_SENSORS)
        ]
        for frame_j in range(FIFO_FRAMES)
    ]


def read_due_burst_payload(ser):
    while is_running:
        header = ser.read(1)
        if not header:
            return None
        if header[0] == DUE_FRAME_HEADER:
            break
    else:
        return None

    payload = ser.read(DUE_PAYLOAD_SIZE)
    if len(payload) != DUE_PAYLOAD_SIZE:
        print("DUE: incomplete packet discarded.", file=sys.stderr)
        return None

    footer = ser.read(1)
    if len(footer) != 1:
        print("DUE: incomplete packet footer discarded.", file=sys.stderr)
        return None
    if footer[0] != DUE_FRAME_FOOTER:
        print("DUE: malformed packet footer discarded.", file=sys.stderr)
        return None

    return payload


def due_reader():
    ser = None
    try:
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        ser = serial.Serial(DUE_PORT, DUE_BAUD_RATE, timeout=0.1)
        ser.reset_input_buffer()
        print(f"DUE connected on {DUE_PORT}; waiting for logging start.", file=sys.stderr)
        due_ready.set()
        cleared_at_start = False
        probe_deadline = time.perf_counter() + DUE_PRESTART_PROBE_TIMEOUT
        received_prestart = False
        while is_running and not logging_started.is_set() and time.perf_counter() < probe_deadline:
            payload = read_due_burst_payload(ser)
            if payload is None:
                continue
            print(
                f"DUE receiving valid burst ({len(payload)} bytes); waiting for logging start.",
                file=sys.stderr,
            )
            received_prestart = True
            break
        if is_running and not logging_started.is_set() and not received_prestart:
            print(
                "DUE connected but no valid burst received before start probe timeout.",
                file=sys.stderr,
            )
        while is_running:
            if not logging_started.wait(timeout=0.05):
                continue
            if not is_running:
                break
            if not cleared_at_start:
                ser.reset_input_buffer()
                cleared_at_start = True
            payload = read_due_burst_payload(ser)
            if payload is None:
                continue
            due_queue.put((elapsed_ns(), payload))
    except Exception as exc:
        print(f"DUE error: {exc}", file=sys.stderr)
        reader_errors.put(("DUE", str(exc)))
    finally:
        if ser is not None and ser.is_open:
            ser.close()


def afd50_reader():
    bus = None
    try:
        if can is None:
            raise RuntimeError("python-can is not installed")
        bus = can.interface.Bus(
            interface=CAN_INTERFACE,
            channel=CAN_CHANNEL,
            bitrate=CAN_BITRATE,
        )
        print("AFD50 connected; waiting for logging start.", file=sys.stderr)
        afd_ready.set()
        cleared_at_start = False
        probe_deadline = time.perf_counter() + AFD50_PRESTART_PROBE_TIMEOUT
        received_prestart = False
        while is_running and not logging_started.is_set() and time.perf_counter() < probe_deadline:
            msg = bus.recv(timeout=0.01)
            if msg is None or msg.arbitration_id != AFD50_FORCE_ID:
                continue
            print(
                f"AFD50 receiving force frame id=0x{msg.arbitration_id:03X}; waiting for logging start.",
                file=sys.stderr,
            )
            received_prestart = True
            break
        if is_running and not logging_started.is_set() and not received_prestart:
            print(
                "AFD50 connected but no force frame received before start probe timeout.",
                file=sys.stderr,
            )
        while is_running:
            if not logging_started.wait(timeout=0.05):
                continue
            if not is_running:
                break
            if not cleared_at_start:
                while bus.recv(timeout=0.0) is not None:
                    pass
                cleared_at_start = True
            msg = bus.recv(timeout=0.01)
            if msg is None or msg.arbitration_id != AFD50_FORCE_ID:
                continue
            raw_data = bytes(msg.data)
            if len(raw_data) < 8:
                raw_data = raw_data.ljust(8, b"\x00")
            afd_queue.put((elapsed_ns(), msg.arbitration_id, raw_data[:8]))
    except Exception as exc:
        if afd_ready.is_set():
            print(f"AFD50 disconnected during logging; continuing without AFD50. ({exc})", file=sys.stderr)
        reader_errors.put(("AFD50", str(exc)))
    finally:
        if bus is not None:
            bus.shutdown()


def loadcell_reader():
    ser = None
    try:
        if open_loadcell_serial is None:
            raise RuntimeError("loadcell_bin_logger.py could not be imported")

        class LoadcellArgs:
            port = LOADCELL_PORT
            baud = LOADCELL_BAUD
            timeout = LOADCELL_TIMEOUT
            read_size = LOADCELL_READ_SIZE
            rx_buffer_size = LOADCELL_RX_BUFFER_SIZE
            keep_input_buffer = False

        ser = open_loadcell_serial(LoadcellArgs)
        print(
            f"Loadcell connected on {LOADCELL_PORT}; waiting for logging start.",
            file=sys.stderr,
        )
        loadcell_ready.set()
        cleared_at_start = False
        read_buffer = bytearray(LOADCELL_READ_SIZE)
        read_view = memoryview(read_buffer)
        probe_deadline = time.perf_counter() + LOADCELL_PRESTART_PROBE_TIMEOUT
        received_prestart = False

        while is_running and not logging_started.is_set() and time.perf_counter() < probe_deadline:
            payload_size = ser.readinto(read_buffer)
            if payload_size:
                print(
                    f"Loadcell receiving data ({payload_size} bytes); waiting for logging start.",
                    file=sys.stderr,
                )
                received_prestart = True
                break

        if is_running and not logging_started.is_set() and not received_prestart:
            print(
                "Loadcell connected but no data received before start probe timeout.",
                file=sys.stderr,
            )

        while is_running:
            if not logging_started.wait(timeout=0.05):
                continue
            if not is_running:
                break
            if not cleared_at_start:
                ser.reset_input_buffer()
                cleared_at_start = True

            payload_size = ser.readinto(read_buffer)
            if not payload_size:
                continue
            loadcell_queue.put((elapsed_ns(), bytes(read_view[:payload_size])))
    except Exception as exc:
        if loadcell_ready.is_set():
            print(f"Loadcell disconnected during logging; continuing without loadcell. ({exc})", file=sys.stderr)
        reader_errors.put(("LOADCELL", str(exc)))
    finally:
        if ser is not None and ser.is_open:
            ser.close()


class BinaryBuffer:
    def __init__(self, binfile, record_struct, payload_size=0):
        self.binfile = binfile
        self.record_struct = record_struct
        self.payload_size = payload_size
        self.record_size = record_struct.size + payload_size
        self.buffer = bytearray(self.record_size * BUFFER_RECORDS)
        self.offset = 0
        self.count = 0

    def write_packed(self, *values):
        self._ensure_space()
        self.record_struct.pack_into(self.buffer, self.offset, *values)
        self.offset += self.record_struct.size
        self.count += 1

    def write_due_payload(self, sample_elapsed_ns, payload):
        self._ensure_space()
        self.record_struct.pack_into(self.buffer, self.offset, sample_elapsed_ns)
        self.offset += self.record_struct.size
        self.buffer[self.offset:self.offset + self.payload_size] = payload
        self.offset += self.payload_size
        self.count += 1

    def write_variable_payload(self, sample_elapsed_ns, payload):
        payload_size = len(payload)
        record_size = self.record_struct.size + payload_size
        if record_size > len(self.buffer):
            self.flush()
            self.binfile.write(self.record_struct.pack(sample_elapsed_ns, payload_size))
            self.binfile.write(payload)
            self.count += 1
            return
        if self.offset + record_size > len(self.buffer):
            self.flush()
        self.record_struct.pack_into(self.buffer, self.offset, sample_elapsed_ns, payload_size)
        self.offset += self.record_struct.size
        self.buffer[self.offset:self.offset + payload_size] = payload
        self.offset += payload_size
        self.count += 1

    def _ensure_space(self):
        if self.offset + self.record_size > len(self.buffer):
            self.flush()

    def flush(self):
        if self.offset:
            self.binfile.write(self.buffer[:self.offset])
            self.offset = 0


class NMC_AXES_EXPR(Structure):
    _pack_ = 1
    _fields_ = [
        ("nBusy", c_short * 8),
        ("nError", c_short * 8),
        ("nNear", c_short * 8),
        ("nPLimit", c_short * 8),
        ("nMLimit", c_short * 8),
        ("nAlarm", c_short * 8),
        ("nEmer", c_short * 2),
        ("nSwPLimit", c_short * 8),
        ("nInpo", c_short * 8),
        ("nHome", c_short * 8),
        ("nEncZ", c_short * 8),
        ("nOrg", c_short * 8),
        ("nSReady", c_short * 8),
        ("nContStatus", c_short * 2),
        ("nDummy", c_short * 6),
        ("nSwMLimit", c_short * 8),
        ("lEnc", c_int * 8),
        ("lCmd", c_int * 8),
        ("dEnc", c_double * 8),
        ("dCmd", c_double * 8),
        ("dummy", c_char * 4),
    ]


def configure_nmc_functions(nmc):
    nmc.nmc_OpenDeviceEx.argtypes = [c_short, c_int]
    nmc.nmc_OpenDeviceEx.restype = c_short
    nmc.nmc_CloseDevice.argtypes = [c_short]
    nmc.nmc_CloseDevice.restype = c_short
    nmc.nmc_GetAxesExpress.argtypes = [c_short, POINTER(NMC_AXES_EXPR)]
    nmc.nmc_GetAxesExpress.restype = c_short


def write_json_header(binfile, magic, header):
    binfile.write(magic.encode("ascii") + b"\n")
    binfile.write(json.dumps(header, separators=(",", ":")).encode("ascii"))
    binfile.write(b"\nEND_HEADER\n")


def write_due_header(binfile):
    write_json_header(
        binfile,
        "DUE_RAW_BURST_BIN_V1",
        {
            "format": "DUE_RAW_BURST_BIN_V1",
            "record_struct": "<Q + 640 raw payload bytes",
            "record_bytes": DUE_RECORD_STRUCT.size + DUE_PAYLOAD_SIZE,
            "columns": ["elapsed_ns", "payload"],
            "payload_layout": "sensor-major uint32 little-endian: sensor[16][frame10]",
            "num_sensors": NUM_SENSORS,
            "fifo_frames": FIFO_FRAMES,
            "baud_rate": DUE_BAUD_RATE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def write_afd50_header(binfile):
    write_json_header(
        binfile,
        "AFD50_CAN_RAW_BIN_V1",
        {
            "format": "AFD50_CAN_RAW_BIN_V1",
            "record_struct": "<QH8s",
            "record_bytes": AFD50_RECORD_STRUCT.size,
            "columns": ["elapsed_ns", "arbitration_id", "data"],
            "force_id": AFD50_FORCE_ID,
            "can_interface": CAN_INTERFACE,
            "can_channel": CAN_CHANNEL,
            "can_bitrate": CAN_BITRATE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def write_ethermotion_header(binfile, dev_no):
    write_json_header(
        binfile,
        "ETHERMOTION_ENCODER_BIN_V1",
        {
            "format": "ETHERMOTION_ENCODER_BIN_V1",
            "record_struct": "<Qdddd",
            "record_bytes": ETHERMOTION_RECORD_STRUCT.size,
            "columns": ["elapsed_ns", "x_cmd", "y_cmd", "z_cmd", "u_cmd"],
            "units": {
                "elapsed_ns": "nanoseconds since logging start",
                "x_cmd": "controller command position unit",
                "y_cmd": "controller command position unit",
                "z_cmd": "controller command position unit",
                "u_cmd": "controller command position unit",
            },
            "dev_no": dev_no,
            "axes": ETHERMOTION_AXES,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def write_loadcell_header(binfile):
    write_json_header(
        binfile,
        LOADCELL_MAGIC,
        {
            "format": LOADCELL_MAGIC,
            "record_struct": "<QI + payload",
            "record_bytes": "variable",
            "columns": ["elapsed_ns", "payload_size", "payload"],
            "port": LOADCELL_PORT,
            "baud": LOADCELL_BAUD,
            "bytesize": 8,
            "parity": "N",
            "stopbits": 1,
            "timeout_sec": LOADCELL_TIMEOUT,
            "read_size": LOADCELL_READ_SIZE,
            "rx_buffer_size": LOADCELL_RX_BUFFER_SIZE,
            "timebase": "nanoseconds since integrated logging start",
            "payload": "raw RS232 bytes from indicator",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def drain_due_queue(writer):
    wrote = 0
    while True:
        try:
            sample_elapsed_ns, payload = due_queue.get_nowait()
        except Empty:
            return wrote
        writer.write_due_payload(sample_elapsed_ns, payload)
        wrote += 1


def drain_afd_queue(writer):
    wrote = 0
    while True:
        try:
            sample_elapsed_ns, arbitration_id, raw_data = afd_queue.get_nowait()
        except Empty:
            return wrote
        writer.write_packed(sample_elapsed_ns, arbitration_id, raw_data)
        wrote += 1


def drain_loadcell_queue(writer):
    wrote = 0
    payload_bytes = 0
    while True:
        try:
            sample_elapsed_ns, payload = loadcell_queue.get_nowait()
        except Empty:
            return wrote, payload_bytes
        writer.write_variable_payload(sample_elapsed_ns, payload)
        wrote += 1
        payload_bytes += len(payload)


def wait_for_readers(timeout_s):
    deadline = time.perf_counter() + timeout_s
    afd_reported_unavailable = False
    loadcell_reported_unavailable = False
    while time.perf_counter() < deadline:
        try:
            name, message = reader_errors.get_nowait()
        except Empty:
            pass
        else:
            if name == "DUE":
                raise RuntimeError(f"{name} initialization failed: {message}")
            if name == "AFD50" and not afd_reported_unavailable:
                print("AFD50 disconnected; AFD logging disabled.", file=sys.stderr)
                afd_reported_unavailable = True
            if name == "LOADCELL" and not loadcell_reported_unavailable:
                print("Loadcell disconnected; loadcell logging disabled.", file=sys.stderr)
                loadcell_reported_unavailable = True
        if (
            due_ready.is_set()
            and (afd_ready.is_set() or afd_reported_unavailable)
            and (loadcell_ready.is_set() or loadcell_reported_unavailable)
        ):
            return True
        time.sleep(0.05)

    while True:
        try:
            name, message = reader_errors.get_nowait()
        except Empty:
            break
        if name == "DUE":
            raise RuntimeError(f"{name} initialization failed: {message}")
        if name == "AFD50" and not afd_reported_unavailable:
            print("AFD50 disconnected; AFD logging disabled.", file=sys.stderr)
            afd_reported_unavailable = True
        if name == "LOADCELL" and not loadcell_reported_unavailable:
            print("Loadcell disconnected; loadcell logging disabled.", file=sys.stderr)
            loadcell_reported_unavailable = True

    if due_ready.is_set():
        if not afd_ready.is_set() and not afd_reported_unavailable:
            print("AFD50 disconnected; AFD logging disabled.", file=sys.stderr)
        if not loadcell_ready.is_set() and not loadcell_reported_unavailable:
            print("Loadcell disconnected; loadcell logging disabled.", file=sys.stderr)
        return True

    raise TimeoutError("Timed out waiting for DUE initialization.")


def wait_for_ethermotion_start(nmc, dev_no, group_no, timeout_s):
    axes_status = NMC_AXES_EXPR()
    deadline = None if timeout_s <= 0 else time.perf_counter() + timeout_s
    saw_idle = False

    print(
        "EtherMotion waiting for external node start.",
        file=sys.stderr,
    )
    while is_running:
        ret = nmc.nmc_GetAxesExpress(dev_no, byref(axes_status))
        if ret == 0:
            is_conti_running = axes_status.nContStatus[group_no] != 0
            if not is_conti_running:
                saw_idle = True
            elif saw_idle:
                return
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Timed out waiting for EtherMotion node start.")
        time.sleep(0.001)

    raise RuntimeError("Stopped before EtherMotion node start was detected.")


def wait_for_manual_start():
    print("Manual start mode. Press Enter when the EtherMotion node start command is issued.", file=sys.stderr)
    input()


def _self_check_due_parser():
    values = [
        sensor_i * 1000 + frame_j
        for sensor_i in range(NUM_SENSORS)
        for frame_j in range(FIFO_FRAMES)
    ]
    payload = struct.pack(DUE_PACKET_FORMAT, *values)
    rows = payload_to_rows(payload)

    assert len(rows) == FIFO_FRAMES
    assert all(len(row) == NUM_SENSORS for row in rows)
    assert rows[0] == [sensor_i * 1000 for sensor_i in range(NUM_SENSORS)]
    assert rows[9] == [sensor_i * 1000 + 9 for sensor_i in range(NUM_SENSORS)]

    class FakeSerial:
        def __init__(self, data):
            self.data = bytearray(data)

        def read(self, size):
            chunk = self.data[:size]
            del self.data[:size]
            return bytes(chunk)

    good_packet = bytes([0x00, DUE_FRAME_HEADER]) + payload + bytes([DUE_FRAME_FOOTER])
    assert read_due_burst_payload(FakeSerial(good_packet)) == payload

    bad_packet = bytes([DUE_FRAME_HEADER]) + payload + bytes([0x00])
    assert read_due_burst_payload(FakeSerial(bad_packet)) is None
    print("DUE parser self-check passed.")


def make_output_paths():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "due": os.path.join(dirs["due"], f"due_raw_burst_{ts}.bin"),
        "afd": os.path.join(dirs["afd"], f"afd50_can_raw_{ts}.bin"),
        "stage": os.path.join(dirs["stage"], f"ethermotion_encoder_{ts}.bin"),
        "loadcell": os.path.join(dirs["loadcell"], f"loadcell_raw_{ts}.bin"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Synchronized binary logging for DUE, AFD50, loadcell, and EtherMotion encoders."
    )
    parser.add_argument("--dev-no", type=int, default=DEV_NO, help=f"EtherMotion device number. Default: {DEV_NO}.")
    parser.add_argument("--group-no", type=int, default=GROUP_NO, help=f"EtherMotion Conti group number. Default: {GROUP_NO}.")
    parser.add_argument(
        "--start-trigger",
        choices=("ethermotion", "manual", "immediate"),
        default="ethermotion",
        help="How to release logging after initialization. Default: ethermotion.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=READY_TIMEOUT,
        help=f"Seconds to wait for DUE and AFD50 initialization. Default: {READY_TIMEOUT}.",
    )
    parser.add_argument(
        "--start-timeout",
        type=float,
        default=START_TRIGGER_TIMEOUT,
        help="Seconds to wait for the start trigger; 0 means wait forever. Default: 0.",
    )
    parser.add_argument(
        "--duration-limit",
        type=float,
        default=DURATION_LIMIT,
        help="Safety duration limit in seconds. Default: 3 hours.",
    )
    parser.add_argument(
        "--ethermotion-idle-timeout",
        type=float,
        default=ETHERMOTION_IDLE_TIMEOUT,
        help="Stop after EtherMotion encoder data is missing or unchanged for this many seconds. Default: 10.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global is_running, log_start_ns
    args = parse_args(argv)
    is_running = True
    due_ready.clear()
    afd_ready.clear()
    loadcell_ready.clear()
    logging_started.clear()
    return_code = 0

    nmc = None
    if os.path.exists(DLL_PATH):
        try:
            nmc = WinDLL(DLL_PATH)
            configure_nmc_functions(nmc)
            ret = nmc.nmc_OpenDeviceEx(args.dev_no, 100)
            if ret != 0:
                print(f"EtherMotion: open failed, ret={ret}.", file=sys.stderr)
                nmc = None
            else:
                print("EtherMotion: connected.", file=sys.stderr)
        except Exception as exc:
            print(f"EtherMotion: DLL load/open failed: {exc}", file=sys.stderr)
            nmc = None
    else:
        print(f"EtherMotion: DLL not found: {DLL_PATH}", file=sys.stderr)

    paths = make_output_paths()
    print(f"DUE binary: {paths['due']}", file=sys.stderr)
    print(f"AFD50 binary: {paths['afd']}", file=sys.stderr)
    print(f"Loadcell binary: {paths['loadcell']}", file=sys.stderr)
    if nmc:
        print(f"EtherMotion binary: {paths['stage']}", file=sys.stderr)

    due_thread = threading.Thread(target=due_reader, daemon=True)
    afd_thread = threading.Thread(target=afd50_reader, daemon=True)
    loadcell_thread = threading.Thread(target=loadcell_reader, daemon=True)
    due_thread.start()
    afd_thread.start()
    loadcell_thread.start()

    try:
        wait_for_readers(args.ready_timeout)
    except Exception as exc:
        print(f"Reader initialization failed: {exc}", file=sys.stderr)
        is_running = False
        logging_started.set()
        due_thread.join(timeout=1.0)
        afd_thread.join(timeout=1.0)
        loadcell_thread.join(timeout=1.0)
        if nmc:
            nmc.nmc_CloseDevice(args.dev_no)
        return 1
    print(
        "DUE connected/waiting; "
        f"AFD50 {'connected/waiting' if afd_ready.is_set() else 'disconnected'}; "
        f"Loadcell {'connected/waiting' if loadcell_ready.is_set() else 'disconnected'}.",
        file=sys.stderr,
    )

    due_count = 0
    afd_count = 0
    loadcell_count = 0
    loadcell_bytes = 0
    stage_count = 0
    stage_errors = 0
    last_report_ns = 0
    last_stage_data_time = None
    ethermotion_stopped_since = None

    f_due = None
    f_afd = None
    f_loadcell = None
    f_stage = None
    due_writer = None
    afd_writer = None
    loadcell_writer = None
    stage_writer = None

    try:
        f_due = open(paths["due"], "wb", buffering=1024 * 1024)
        f_afd = open(paths["afd"], "wb", buffering=1024 * 1024)
        f_loadcell = open(paths["loadcell"], "wb", buffering=1024 * 1024)
        write_due_header(f_due)
        write_afd50_header(f_afd)
        write_loadcell_header(f_loadcell)
        due_writer = BinaryBuffer(f_due, DUE_RECORD_STRUCT, DUE_PAYLOAD_SIZE)
        afd_writer = BinaryBuffer(f_afd, AFD50_RECORD_STRUCT)
        loadcell_writer = BinaryBuffer(f_loadcell, LOADCELL_RECORD_STRUCT)

        if nmc:
            f_stage = open(paths["stage"], "wb", buffering=1024 * 1024)
            write_ethermotion_header(f_stage, args.dev_no)
            stage_writer = BinaryBuffer(f_stage, ETHERMOTION_RECORD_STRUCT)

        if args.start_trigger == "ethermotion":
            if not nmc:
                raise RuntimeError("EtherMotion start trigger requested, but EtherMotion is not connected.")
            wait_for_ethermotion_start(nmc, args.dev_no, args.group_no, args.start_timeout)
        elif args.start_trigger == "manual":
            wait_for_manual_start()
        else:
            print("Immediate start trigger selected.", file=sys.stderr)

        log_start_ns = time.perf_counter_ns()
        logging_started.set()
        print("EtherMotion running detected. Logging started.", file=sys.stderr)
        run_start = time.perf_counter()
        last_report_ns = log_start_ns
        last_stage_data_time = run_start
        print("logging...", file=sys.stderr)

        while True:
            loop_start = time.perf_counter()
            now_ns = time.perf_counter_ns()

            if loop_start - run_start >= args.duration_limit:
                print("Duration limit reached; stopping.", file=sys.stderr)
                break

            if nmc:
                try:
                    axes_status = NMC_AXES_EXPR()
                    axes_ret = nmc.nmc_GetAxesExpress(args.dev_no, byref(axes_status))
                    if axes_ret == 0:
                        position = (
                            axes_status.dCmd[0],
                            axes_status.dCmd[1],
                            axes_status.dCmd[2],
                            axes_status.dCmd[3],
                        )
                        stage_writer.write_packed(
                            elapsed_ns(),
                            position[0],
                            position[1],
                            position[2],
                            position[3],
                        )
                        stage_count += 1
                        last_stage_data_time = loop_start
                        is_conti_running = axes_status.nContStatus[args.group_no] != 0
                        is_axis_busy = any(axes_status.nBusy[axis] != 0 for axis in ETHERMOTION_AXES)
                        if is_conti_running or is_axis_busy:
                            ethermotion_stopped_since = None
                        elif ethermotion_stopped_since is None:
                            ethermotion_stopped_since = loop_start
                    else:
                        stage_errors += 1
                except Exception:
                    stage_errors += 1

                if args.ethermotion_idle_timeout > 0:
                    no_data_for = loop_start - last_stage_data_time
                    if no_data_for >= args.ethermotion_idle_timeout:
                        print(
                            f"EtherMotion data stopped for {no_data_for:.1f}s; stopping.",
                            file=sys.stderr,
                        )
                        break
                    if (
                        ethermotion_stopped_since is not None
                        and loop_start - ethermotion_stopped_since >= args.ethermotion_idle_timeout
                    ):
                        print(
                            f"EtherMotion motion stopped for {loop_start - ethermotion_stopped_since:.1f}s; stopping.",
                            file=sys.stderr,
                        )
                        break

            due_count += drain_due_queue(due_writer)
            afd_count += drain_afd_queue(afd_writer)
            drained_loadcell_count, drained_loadcell_bytes = drain_loadcell_queue(loadcell_writer)
            loadcell_count += drained_loadcell_count
            loadcell_bytes += drained_loadcell_bytes

            if now_ns - last_report_ns >= 1_000_000_000:
                due_writer.flush()
                afd_writer.flush()
                loadcell_writer.flush()
                if stage_writer:
                    stage_writer.flush()
                last_report_ns = now_ns

            sleep_time = POLLING_INTERVAL - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStop requested by user.", file=sys.stderr)
        return_code = 130
    except Exception as exc:
        print(f"Integrated logger error: {exc}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        is_running = False
        logging_started.set()
        due_thread.join(timeout=1.0)
        afd_thread.join(timeout=1.0)
        loadcell_thread.join(timeout=1.0)
        try:
            if due_writer is not None:
                due_count += drain_due_queue(due_writer)
                due_writer.flush()
            if afd_writer is not None:
                afd_count += drain_afd_queue(afd_writer)
                afd_writer.flush()
            if loadcell_writer is not None:
                drained_loadcell_count, drained_loadcell_bytes = drain_loadcell_queue(loadcell_writer)
                loadcell_count += drained_loadcell_count
                loadcell_bytes += drained_loadcell_bytes
                loadcell_writer.flush()
            if stage_writer is not None:
                stage_writer.flush()
            if f_due is not None:
                f_due.flush()
                f_due.close()
            if f_afd is not None:
                f_afd.flush()
                f_afd.close()
            if f_loadcell is not None:
                f_loadcell.flush()
                f_loadcell.close()
            if f_stage is not None:
                f_stage.flush()
                f_stage.close()
        finally:
            if nmc:
                nmc.nmc_CloseDevice(args.dev_no)
        print(
            f"Saved binary logs. due_bursts={due_count}, afd={afd_count}, "
            f"loadcell_chunks={loadcell_count}, loadcell_bytes={loadcell_bytes}, "
            f"ethermotion={stage_count}, ethermotion_errors={stage_errors}",
            file=sys.stderr,
        )
    return return_code


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        log_start_ns = time.perf_counter_ns()
        _self_check_due_parser()
    else:
        raise SystemExit(main())
