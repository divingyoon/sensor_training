import sys
import can
import time
import csv
from datetime import datetime

# --- 사용자 설정 ---
CAN_INTERFACE = 'ixxat'
CAN_CHANNEL = 0
CAN_BITRATE = 1000000
FORCE_SENSOR_CAN_ID = 0x1A
TORQUE_SENSOR_CAN_ID = 0x1B
CAL_SAMPLES = 100

# --- 데이터 저장 설정 ---
LOG_DIR = "C:\\Users\\SORO2\\Desktop\\skin_ws\\acc_v2\\logs"

# --- 데이터 변환 함수 ---
def to_u16(buf, idx):
    return (buf[idx * 2] << 8) | buf[idx * 2 + 1]

def conv_f(raw):
    return raw / 300.0 - 100.0

def conv_t(raw):
    return raw / 50000.0 - 0.6

def main(fx_thresh, fy_thresh, fz_thresh):
    bus = None
    while True:
        try:
            if bus is None:
                print("[AFD50] CAN 버스 연결 시도 중...", file=sys.stderr)
                bus = can.interface.Bus(interface=CAN_INTERFACE, channel=CAN_CHANNEL, bitrate=CAN_BITRATE)
                print("[AFD50] CAN 버스 연결 성공.", file=sys.stderr)
                break
        except can.CanError:
            print(f"[AFD50] CAN 버스 연결 실패. 5초 후 재시도...", file=sys.stderr)
            time.sleep(5)

    # --- 바이어스 보정 ---
    print("[AFD50] 바이어스 보정 시작...", file=sys.stderr)
    cnt_f, cnt_t = 0, 0
    temp_bias_f = [0.0, 0.0, 0.0]
    temp_bias_t = [0.0, 0.0, 0.0]
    
    start_time = time.time()
    while (cnt_f < CAL_SAMPLES or cnt_t < CAL_SAMPLES) and (time.time() - start_time < 10):
        msg = bus.recv(timeout=0.1)
        if msg is None: continue

        if msg.arbitration_id == FORCE_SENSOR_CAN_ID and cnt_f < CAL_SAMPLES:
            r = [to_u16(msg.data, i) for i in range(3)]
            f = [conv_f(val) for val in r]
            for i in range(3): temp_bias_f[i] += f[i]
            cnt_f += 1
        elif msg.arbitration_id == TORQUE_SENSOR_CAN_ID and cnt_t < CAL_SAMPLES:
            r = [to_u16(msg.data, i) for i in range(3)]
            t = [conv_t(val) for val in r]
            for i in range(3): temp_bias_t[i] += t[i]
            cnt_t += 1

    bias_f = [val / cnt_f if cnt_f > 0 else 0.0 for val in temp_bias_f]
    bias_t = [val / cnt_t if cnt_t > 0 else 0.0 for val in temp_bias_t]
    print(f"[AFD50] 바이어스 보정 완료.", file=sys.stderr)
    print(f"[AFD50] Force Bias: Fx={bias_f[0]:.3f}, Fy={bias_f[1]:.3f}, Fz={bias_f[2]:.3f}", file=sys.stderr)
    print(f"[AFD50] Torque Bias: Tx={bias_t[0]:.3f}, Ty={bias_t[1]:.3f}, Tz={bias_t[2]:.3f}", file=sys.stderr)

    # --- 로그 파일 설정 ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{LOG_DIR}\\afd50_log_{timestamp_str}.csv"
    header = ['timestamp', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    try:
        with open(log_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            print(f"[AFD50] Logging data to {log_filename}", file=sys.stderr)
            print(f"[AFD50] Stop thresholds: Fx>{fx_thresh}, Fy>{fy_thresh}, Fz>{fz_thresh}", file=sys.stderr)

            latest_force = [0.0] * 3
            latest_torque = [0.0] * 3

            # --- 메시지 수신 및 처리 루프 ---
            for msg in bus:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                if msg.arbitration_id == FORCE_SENSOR_CAN_ID:
                    r = [to_u16(msg.data, i) for i in range(3)]
                    f = [conv_f(val) for val in r]
                    latest_force[0] = f[0] - bias_f[0]
                    latest_force[1] = f[1] - bias_f[1]
                    latest_force[2] = -(f[2] - bias_f[2])
                elif msg.arbitration_id == TORQUE_SENSOR_CAN_ID:
                    r = [to_u16(msg.data, i) for i in range(3)]
                    t = [conv_t(val) for val in r]
                    latest_torque = [val - bias for val, bias in zip(t, bias_t)]
                else:
                    continue

                all_values = latest_force + latest_torque
                
                # 1. CSV에 데이터 로깅
                writer.writerow([ts] + [f"{v:.4f}" for v in all_values])

                # 2. 임계값 확인 및 정지 신호 전송 (Force만)
                if abs(all_values[0]) > fx_thresh or abs(all_values[1]) > fy_thresh or abs(all_values[2]) > fz_thresh:
                    print("STOP_AFD50", flush=True)

    except can.CanError as e:
        print(f"[AFD50] CAN Error: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("[AFD50] Process interrupted by user.", file=sys.stderr)
    finally:
        bus.shutdown()
        print("[AFD50] CAN bus shut down.", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python afd50_reader_csv.py <fx_thresh> <fy_thresh> <fz_thresh>", file=sys.stderr)
        sys.exit(1)

    fx_threshold = float(sys.argv[1])
    fy_threshold = float(sys.argv[2])
    fz_threshold = float(sys.argv[3])

    import os
    os.makedirs(LOG_DIR, exist_ok=True)
    
    main(fx_threshold, fy_threshold, fz_threshold)