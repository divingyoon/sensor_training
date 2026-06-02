import argparse
import csv
import json
import os
import re
import shutil
import struct
import sys
from datetime import datetime
from pathlib import Path


NUM_SENSORS = 16
FIFO_FRAMES = 10
DUE_PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4
DUE_RECORD_STRUCT = struct.Struct("<Q")
DUE_PAYLOAD_STRUCT = struct.Struct("<" + ("I" * NUM_SENSORS * FIFO_FRAMES))
AFD50_RECORD_STRUCT = struct.Struct("<QH8s")
ETHERMOTION_RECORD_STRUCT_3AXIS = struct.Struct("<Qddd")
ETHERMOTION_RECORD_STRUCT_4AXIS = struct.Struct("<Qdddd")
LOADCELL_RECORD_STRUCT = struct.Struct("<QI")  # elapsed_ns, payload_size

DUE_MAGIC = "DUE_RAW_BURST_BIN_V1"
AFD50_MAGIC = "AFD50_CAN_RAW_BIN_V1"
ETHERMOTION_MAGIC = "ETHERMOTION_ENCODER_BIN_V1"
LOADCELL_MAGIC = "LOADCELL_BIN_V1"
AFD50_FORCE_ID = 0x01A


def read_bin_header(binfile):
    magic = binfile.readline().decode("ascii", errors="replace").strip()
    if not magic:
        raise ValueError("missing binary magic header")

    header_line = binfile.readline()
    if not header_line:
        raise ValueError(f"{magic}: missing JSON header")

    try:
        header = json.loads(header_line.decode("ascii"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{magic}: invalid JSON header: {exc}") from exc

    end_header = binfile.readline()
    if end_header != b"END_HEADER\n":
        raise ValueError(f"{magic}: missing END_HEADER marker")

    return magic, header


def payload_to_rows(payload):
    values = DUE_PAYLOAD_STRUCT.unpack(payload)
    return [
        [
            values[sensor_i * FIFO_FRAMES + frame_i]
            for sensor_i in range(NUM_SENSORS)
        ]
        for frame_i in range(FIFO_FRAMES)
    ]


def iter_due_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != DUE_MAGIC:
            raise ValueError(f"{path}: expected {DUE_MAGIC}, got {magic}")

        record_bytes = int(header.get("record_bytes", DUE_RECORD_STRUCT.size + DUE_PAYLOAD_SIZE))
        expected_bytes = DUE_RECORD_STRUCT.size + DUE_PAYLOAD_SIZE
        if record_bytes != expected_bytes:
            raise ValueError(f"{path}: unsupported DUE record size {record_bytes}")

        record_i = 0
        while True:
            elapsed_bytes = binfile.read(DUE_RECORD_STRUCT.size)
            if not elapsed_bytes:
                break
            if len(elapsed_bytes) != DUE_RECORD_STRUCT.size:
                raise ValueError(f"{path}: truncated DUE elapsed_ns at record {record_i}")

            payload = binfile.read(DUE_PAYLOAD_SIZE)
            if len(payload) != DUE_PAYLOAD_SIZE:
                raise ValueError(f"{path}: truncated DUE payload at record {record_i}")

            elapsed_ns = DUE_RECORD_STRUCT.unpack(elapsed_bytes)[0]
            yield record_i, elapsed_ns, payload
            record_i += 1


def convert_due_bin(input_path, output_path):
    row_count = 0
    burst_count = 0
    headers = ["elapsed_ns", "time_s", "burst_index", "frame_index"]
    headers.extend([f"Skin{i}" for i in range(1, NUM_SENSORS + 1)])

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for burst_i, elapsed_ns, payload in iter_due_records(input_path):
            burst_count += 1
            for frame_i, row in enumerate(payload_to_rows(payload)):
                writer.writerow([elapsed_ns, elapsed_ns / 1_000_000_000, burst_i, frame_i] + row)
                row_count += 1

    return {"bursts": burst_count, "rows": row_count}


def iter_ethermotion_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != ETHERMOTION_MAGIC:
            raise ValueError(f"{path}: expected {ETHERMOTION_MAGIC}, got {magic}")

        record_bytes = int(header.get("record_bytes", ETHERMOTION_RECORD_STRUCT_3AXIS.size))
        if record_bytes == ETHERMOTION_RECORD_STRUCT_3AXIS.size:
            record_struct = ETHERMOTION_RECORD_STRUCT_3AXIS
            axis_count = 3
        elif record_bytes == ETHERMOTION_RECORD_STRUCT_4AXIS.size:
            record_struct = ETHERMOTION_RECORD_STRUCT_4AXIS
            axis_count = 4
        else:
            raise ValueError(f"{path}: unsupported EtherMotion record size {record_bytes}")

        record_i = 0
        while True:
            record = binfile.read(record_struct.size)
            if not record:
                break
            if len(record) != record_struct.size:
                raise ValueError(f"{path}: truncated EtherMotion record {record_i}")
            yield record_i, axis_count, record_struct.unpack(record)
            record_i += 1


def convert_ethermotion_bin(input_path, output_path):
    row_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header_written = False
        axis_count_out = None
        for _, axis_count, record_values in iter_ethermotion_records(input_path):
            if not header_written:
                if axis_count == 4:
                    axis_headers = ["X", "Y", "Z", "U"]
                else:
                    axis_headers = ["X", "Y", "Z"]
                writer.writerow(["elapsed_ns", "time_s"] + axis_headers)
                axis_count_out = axis_count
                header_written = True
            elapsed_ns = record_values[0]
            writer.writerow([elapsed_ns, elapsed_ns / 1_000_000_000] + list(record_values[1:]))
            row_count += 1
        if not header_written:
            writer.writerow(["elapsed_ns", "time_s", "X", "Y", "Z"])
            axis_count_out = 3
    return {"rows": row_count, "axes": axis_count_out}


def to_u16_be(data, index):
    return (data[index * 2] << 8) | data[index * 2 + 1]


def force_from_raw(raw):
    return raw / 300.0 - 100.0


def iter_afd50_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != AFD50_MAGIC:
            raise ValueError(f"{path}: expected {AFD50_MAGIC}, got {magic}")

        record_bytes = int(header.get("record_bytes", AFD50_RECORD_STRUCT.size))
        if record_bytes != AFD50_RECORD_STRUCT.size:
            raise ValueError(f"{path}: unsupported AFD50 record size {record_bytes}")

        record_i = 0
        while True:
            record = binfile.read(AFD50_RECORD_STRUCT.size)
            if not record:
                break
            if len(record) != AFD50_RECORD_STRUCT.size:
                raise ValueError(f"{path}: truncated AFD50 record {record_i}")
            yield record_i, AFD50_RECORD_STRUCT.unpack(record)
            record_i += 1


def afd50_force_rows(input_path):
    rows = []
    for record_i, (elapsed_ns, arbitration_id, raw_data) in iter_afd50_records(input_path):
        if arbitration_id != AFD50_FORCE_ID:
            continue
        raw_fx = to_u16_be(raw_data, 0)
        raw_fy = to_u16_be(raw_data, 1)
        raw_fz = to_u16_be(raw_data, 2)
        rows.append(
            {
                "record_index": record_i,
                "elapsed_ns": elapsed_ns,
                "arbitration_id": arbitration_id,
                "raw_hex": raw_data.hex(),
                "raw_fx": raw_fx,
                "raw_fy": raw_fy,
                "raw_fz": raw_fz,
                "fx_unbiased": force_from_raw(raw_fx),
                "fy_unbiased": force_from_raw(raw_fy),
                "fz_unbiased": force_from_raw(raw_fz),
            }
        )
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def calculate_afd50_bias(rows, bias_samples):
    sample_rows = rows[:bias_samples] if bias_samples > 0 else []
    return {
        "fx": mean([row["fx_unbiased"] for row in sample_rows]),
        "fy": mean([row["fy_unbiased"] for row in sample_rows]),
        "fz": mean([row["fz_unbiased"] for row in sample_rows]),
        "samples": len(sample_rows),
    }


def convert_afd50_bin(input_path, output_path, bias_samples=200, invert_fz=True):
    rows = afd50_force_rows(input_path)
    bias = calculate_afd50_bias(rows, bias_samples)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "elapsed_ns",
                "time_s",
                "arbitration_id",
                "raw_hex",
                "raw_fx",
                "raw_fy",
                "raw_fz",
                "Fx",
                "Fy",
                "Fz",
            ]
        )

        for row in rows:
            fx = row["fx_unbiased"] - bias["fx"]
            fy = row["fy_unbiased"] - bias["fy"]
            fz = row["fz_unbiased"] - bias["fz"]
            if invert_fz:
                fz = -fz

            writer.writerow(
                [
                    row["elapsed_ns"],
                    row["elapsed_ns"] / 1_000_000_000,
                    f"0x{row['arbitration_id']:03X}",
                    row["raw_hex"],
                    row["raw_fx"],
                    row["raw_fy"],
                    row["raw_fz"],
                    f"{fx:.6f}",
                    f"{fy:.6f}",
                    f"{fz:.6f}",
                ]
            )

    return {"rows": len(rows), "bias": bias, "invert_fz": invert_fz}


def iter_loadcell_records(path):
    with open(path, "rb") as binfile:
        magic, header = read_bin_header(binfile)
        if magic != LOADCELL_MAGIC:
            raise ValueError(f"{path}: expected {LOADCELL_MAGIC}, got {magic}")

        record_i = 0
        while True:
            record_header = binfile.read(LOADCELL_RECORD_STRUCT.size)
            if not record_header:
                break
            if len(record_header) != LOADCELL_RECORD_STRUCT.size:
                raise ValueError(f"{path}: truncated record header at record {record_i}")

            elapsed_ns, payload_size = LOADCELL_RECORD_STRUCT.unpack(record_header)
            payload = binfile.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError(f"{path}: truncated payload at record {record_i}")

            yield record_i, elapsed_ns, payload
            record_i += 1


def convert_loadcell_bin(input_path, output_path):
    VALUE_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

    def parse_kg(raw_line):
        match = VALUE_PATTERN.search(raw_line)
        if match is None:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    rows = []
    buffer = b""
    for record_i, elapsed_ns, payload in iter_loadcell_records(input_path):
        buffer += payload
        # Records in the loadcell bin are just raw serial dumps, which are newline-terminated.
        # We need to re-assemble the lines. A single read can have multiple lines, and
        # a line can be split across multiple reads.
        while b"\n" in buffer:
            line, _, buffer = buffer.partition(b"\n")
            line_str = line.decode("ascii", errors="replace").strip()
            if not line_str:
                continue

            kg = parse_kg(line_str)
            if kg is None:
                continue
            rows.append((elapsed_ns, kg))

    if buffer:
        line_str = buffer.decode("ascii", errors="replace").strip()
        if line_str:
            kg = parse_kg(line_str)
            if kg is not None:
                rows.append((elapsed_ns, kg))

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["elapsed_ns", "time_s", "kg"])
        for elapsed_ns, kg in rows:
            writer.writerow([elapsed_ns, elapsed_ns / 1_000_000_000, f"{kg:.6f}"])

    return {"rows": len(rows)}


def extract_timestamp(path):
    match = re.search(r"(\d{8}_\d{6})", path.name)
    if match:
        return match.group(1)
    return path.stat().st_mtime


def sorted_bins(directory, pattern):
    return sorted(Path(directory).glob(pattern), key=lambda path: (extract_timestamp(path), path.name))


def next_output_dir(full_data_dir, date_prefix):
    full_data_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(date_prefix)}_test(\d+)$")
    used_numbers = []
    for child in full_data_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            used_numbers.append(int(match.group(1)))
    next_number = max(used_numbers, default=0) + 1
    return full_data_dir / f"{date_prefix}_test{next_number}"


def discover_bin_sets(base_dir):
    due_bins = sorted_bins(base_dir / "due data", "due_raw_burst_*.bin")
    ethermotion_bins = sorted_bins(base_dir / "ethermotion data", "ethermotion_encoder_*.bin")
    afd_bins = sorted_bins(base_dir / "afd_50 data", "afd50_can_raw_*.bin")
    loadcell_bins = sorted_bins(base_dir / "loadcell data", "loadcell_raw_*.bin")
    set_count = min(len(due_bins), len(ethermotion_bins), len(afd_bins), len(loadcell_bins))
    return [
        {
            "due": due_bins[i],
            "ethermotion": ethermotion_bins[i],
            "afd50": afd_bins[i],
            "loadcell": loadcell_bins[i],
        }
        for i in range(set_count)
    ], {
        "due": len(due_bins),
        "ethermotion": len(ethermotion_bins),
        "afd50": len(afd_bins),
        "loadcell": len(loadcell_bins),
        "matched": set_count,
    }


def write_manifest(output_dir, inputs, stats, args):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "outputs": {
            "due_csv": str(output_dir / "due_data.csv"),
            "ethermotion_csv": str(output_dir / "ethermotion_data.csv"),
            "afd50_csv": str(output_dir / "afd50_data.csv"),
            "loadcell_csv": str(output_dir / "loadcell_data.csv"),
        },
        "conversion": {
            "afd50_force_formula": "raw / 300.0 - 100.0",
            "afd50_bias_samples": args.bias_samples,
            "afd50_invert_fz": not args.no_invert_fz,
            "due_payload_layout": "sensor-major uint32 little-endian: sensor[16][frame10]",
        },
        "stats": stats,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)


def convert_set(inputs, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(inputs["due"], output_dir / inputs["due"].name)
    shutil.copy2(inputs["ethermotion"], output_dir / inputs["ethermotion"].name)
    shutil.copy2(inputs["afd50"], output_dir / inputs["afd50"].name)
    shutil.copy2(inputs["loadcell"], output_dir / inputs["loadcell"].name)

    stats = {}
    print(f"DUE -> {output_dir / 'due_data.csv'}", file=sys.stderr)
    stats["due"] = convert_due_bin(inputs["due"], output_dir / "due_data.csv")

    print(f"EtherMotion -> {output_dir / 'ethermotion_data.csv'}", file=sys.stderr)
    stats["ethermotion"] = convert_ethermotion_bin(inputs["ethermotion"], output_dir / "ethermotion_data.csv")

    print(f"AFD50 -> {output_dir / 'afd50_data.csv'}", file=sys.stderr)
    stats["afd50"] = convert_afd50_bin(
        inputs["afd50"],
        output_dir / "afd50_data.csv",
        bias_samples=args.bias_samples,
        invert_fz=not args.no_invert_fz,
    )

    print(f"Loadcell -> {output_dir / 'loadcell_data.csv'}", file=sys.stderr)
    stats["loadcell"] = convert_loadcell_bin(inputs["loadcell"], output_dir / "loadcell_data.csv")

    write_manifest(output_dir, inputs, stats, args)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert integrated binary logs into per-stream physical/readable CSV files."
    )
    default_base = Path(__file__).resolve().parents[1]
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--all", action="store_true", help="Convert every matched bin trio in sorted order.")
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Matched bin trio index to convert. Default -1 converts the latest matched trio.",
    )
    parser.add_argument(
        "--bias-samples",
        type=int,
        default=200,
        help="Number of initial AFD50 samples used as zero-force bias. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-invert-fz",
        action="store_true",
        help="Do not invert AFD50 Fz after bias removal.",
    )
    parser.add_argument(
        "--date-prefix",
        default=datetime.now().strftime("%Y%m%d"),
        help="Output folder date prefix, e.g. 20260426.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir.resolve()
    full_data_dir = base_dir / "full data"

    bin_sets, counts = discover_bin_sets(base_dir)
    if not bin_sets:
        print(
            "No matched bin sets found. "
            f"Counts: due={counts['due']}, ethermotion={counts['ethermotion']}, "
            f"afd50={counts['afd50']}, loadcell={counts['loadcell']}",
            file=sys.stderr,
        )
        return 1

    if args.all:
        selected_sets = bin_sets
    else:
        try:
            selected_sets = [bin_sets[args.index]]
        except IndexError:
            print(f"Invalid --index {args.index}; matched set count is {len(bin_sets)}.", file=sys.stderr)
            return 1

    print(
        f"Found matched bin sets={counts['matched']} "
        f"(due={counts['due']}, ethermotion={counts['ethermotion']}, "
        f"afd50={counts['afd50']}, loadcell={counts['loadcell']}).",
        file=sys.stderr,
    )

    for inputs in selected_sets:
        output_dir = next_output_dir(full_data_dir, args.date_prefix)
        print(f"Writing converted data to {output_dir}", file=sys.stderr)
        stats = convert_set(inputs, output_dir, args)
        print(
            f"Done: due_rows={stats['due']['rows']}, "
            f"ethermotion_rows={stats['ethermotion']['rows']}, "
            f"afd50_rows={stats['afd50']['rows']}, "
            f"loadcell_rows={stats['loadcell']['rows']}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
