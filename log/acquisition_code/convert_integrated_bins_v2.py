import argparse
import csv
import os
import re
import struct
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Constants matching final_logger_integrated_v2_gui.py
NUM_SENSORS = 16
FIFO_FRAMES = 10
DUE_PAYLOAD_SIZE = NUM_SENSORS * FIFO_FRAMES * 4
AFD50_FORCE_ID = 0x01A

def read_magic(f):
    line = f.readline().decode("ascii", errors="replace").strip()
    return line

def convert_due_v2(input_path, output_path):
    burst_count = 0
    row_count = 0
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "DUE_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")
        
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            headers = ["ns", "time_s", "burst_index", "frame_index"] + [f"S{i+1:02d}" for i in range(NUM_SENSORS)]
            writer.writerow(headers)
            
            record_size = 8 + DUE_PAYLOAD_SIZE # 8 (Q) + 640
            while True:
                data = f.read(record_size)
                if len(data) < record_size: break
                
                ns = struct.unpack("<Q", data[:8])[0]
                payload = data[8:]
                values = struct.unpack("<" + ("I" * NUM_SENSORS * FIFO_FRAMES), payload)
                
                # Split into frames
                for frame_i in range(FIFO_FRAMES):
                    row = [values[sensor_i * FIFO_FRAMES + frame_i] for sensor_i in range(NUM_SENSORS)]
                    writer.writerow([ns, ns / 1e9, burst_count, frame_i] + row)
                    row_count += 1
                burst_count += 1
    return row_count

def convert_em_v2(input_path, output_path):
    row_count = 0
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "EM_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")
            
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ns", "time_s", "X", "Y", "Z", "U"])
            
            record_size = 8 + 32 # 8 (Q) + 4*8 (dddd)
            while True:
                data = f.read(record_size)
                if len(data) < record_size: break
                
                ns, x, y, z, u = struct.unpack("<Qdddd", data)
                writer.writerow([ns, ns / 1e9, x, y, z, u])
                row_count += 1
    return row_count

def convert_afd_v2(input_path, output_path, bias_samples=200):
    rows = []
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "AFD_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")
            
        record_size = 8 + 8 # 8 (Q) + 8 (can data)
        while True:
            data = f.read(record_size)
            if len(data) < record_size: break
            
            ns = struct.unpack("<Q", data[:8])[0]
            raw_data = data[8:]
            
            # Unpack Fx, Fy, Fz (Big Endian U16)
            raw_fx = (raw_data[0] << 8) | raw_data[1]
            raw_fy = (raw_data[2] << 8) | raw_data[3]
            raw_fz = (raw_data[4] << 8) | raw_data[5]
            
            fx_u = raw_fx / 300.0 - 100.0
            fy_u = raw_fy / 300.0 - 100.0
            fz_u = raw_fz / 300.0 - 100.0
            
            rows.append({"ns": ns, "raw_fx": raw_fx, "raw_fy": raw_fy, "raw_fz": raw_fz, "fx_u": fx_u, "fy_u": fy_u, "fz_u": fz_u})

    # Bias calculation
    b_fx = sum(r["fx_u"] for r in rows[:bias_samples]) / max(1, min(len(rows), bias_samples))
    b_fy = sum(r["fy_u"] for r in rows[:bias_samples]) / max(1, min(len(rows), bias_samples))
    b_fz = sum(r["fz_u"] for r in rows[:bias_samples]) / max(1, min(len(rows), bias_samples))

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ns", "time_s", "raw_fx", "raw_fy", "raw_fz", "Fx", "Fy", "Fz"])
        for r in rows:
            fx = r["fx_u"] - b_fx
            fy = r["fy_u"] - b_fy
            fz = -(r["fz_u"] - b_fz) # Invert Fz
            writer.writerow([r["ns"], r["ns"]/1e9, r["raw_fx"], r["raw_fy"], r["raw_fz"], f"{fx:.6f}", f"{fy:.6f}", f"{fz:.6f}"])
    return len(rows)

def convert_lc_v2(input_path, output_path):
    VALUE_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
    row_count = 0
    with open(input_path, "rb") as f:
        magic = read_magic(f)
        if magic != "LC_V2":
            print(f"Warning: {input_path} has unexpected magic {magic}")
            
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ns", "time_s", "kg"])
            
            buffer = b""
            while True:
                header = f.read(12) # 8 (Q) + 4 (I)
                if len(header) < 12: break
                ns, size = struct.unpack("<QI", header)
                payload = f.read(size)
                
                buffer += payload
                while b"\n" in buffer:
                    line, _, buffer = buffer.partition(b"\n")
                    line_str = line.decode("ascii", errors="replace").strip()
                    match = VALUE_PATTERN.search(line_str)
                    if match:
                        kg = float(match.group(0))
                        writer.writerow([ns, ns/1e9, f"{kg:.6f}"])
                        row_count += 1
    return row_count

def discover_sets(base_dir):
    # Try both naming conventions
    due_patterns = ["due_v2_*.bin", "due_raw_burst_*.bin"]
    em_patterns = ["em_v2_*.bin", "ethermotion_encoder_*.bin"]
    lc_patterns = ["lc_v2_*.bin", "loadcell_raw_*.bin"]
    afd_patterns = ["afd_v2_*.bin", "afd50_can_raw_*.bin"]

    def get_files(subdir, patterns):
        found = []
        target_dir = base_dir / subdir
        if not target_dir.exists():
            return []
        for p in patterns:
            found.extend(list(target_dir.glob(p)))
        return sorted(found)

    due_files = get_files("due data", due_patterns)
    em_files = get_files("ethermotion data", em_patterns)
    lc_files = get_files("loadcell data", lc_patterns)
    afd_files = get_files("afd_50 data", afd_patterns)
    
    # Map by timestamp suffix
    def get_ts(p): 
        # Handles YYYYMMDD_HHMMSS.bin
        name = p.name
        match = re.search(r"(\d{8}_\d{6})", name)
        if match:
            return match.group(1)
        return name.split("_")[-1].replace(".bin", "")
    
    sets = {}
    for f in due_files: sets.setdefault(get_ts(f), {})["due"] = f
    for f in em_files: sets.setdefault(get_ts(f), {})["em"] = f
    for f in lc_files: sets.setdefault(get_ts(f), {})["lc"] = f
    for f in afd_files: sets.setdefault(get_ts(f), {})["afd"] = f
    
    # Keep sets that have at least 3 files
    complete = {ts: d for ts, d in sets.items() if len(d) >= 3}
    return complete

def next_output_dir(full_data_dir, date_prefix):
    full_data_dir.mkdir(parents=True, exist_ok=True)
    # Match both date_testN and just testN if needed, but user used date_testN
    pattern = re.compile(rf"^(?:{re.escape(date_prefix)}_)?test(\d+)$")
    used_numbers = []
    for child in full_data_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            used_numbers.append(int(match.group(1)))
    next_number = max(used_numbers, default=0) + 1
    return full_data_dir / f"{date_prefix}_test{next_number}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias-samples", type=int, default=200)
    parser.add_argument("--date-prefix", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--all", action="store_true", help="Process all sets instead of just the latest one")
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parents[1]
    full_data_v2 = base_dir / "full data_v2"
    
    sets = discover_sets(base_dir)
    if not sets:
        print("No complete bin sets found.")
        return

    sorted_ts = sorted(sets.keys())
    
    if args.all:
        process_ts = sorted_ts
    else:
        # Only process the LATEST set
        process_ts = [sorted_ts[-1]]

    for ts in process_ts:
        files = sets[ts]
        out_dir = next_output_dir(full_data_v2, args.date_prefix)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Converting set {ts} -> {out_dir}")

        # Copy original bin files to the output directory
        for p in files.values():
            shutil.copy2(p, out_dir / p.name)
        
        if "due" in files:
            c = convert_due_v2(files["due"], out_dir / "due_data.csv")
            print(f"  DUE: {c} rows")
        if "em" in files:
            c = convert_em_v2(files["em"], out_dir / "ethermotion_data.csv")
            print(f"  EM: {c} rows")
        if "lc" in files:
            c = convert_lc_v2(files["lc"], out_dir / "loadcell_data.csv")
            print(f"  LC: {c} rows")
        if "afd" in files:
            c = convert_afd_v2(files["afd"], out_dir / "afd50_data.csv", bias_samples=args.bias_samples)
            print(f"  AFD: {c} rows")


if __name__ == "__main__":
    main()
