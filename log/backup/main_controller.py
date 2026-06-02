import subprocess
import threading
import serial
import time
import csv
import sys
import os
from datetime import datetime
from queue import Queue, Empty
from collections import deque

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib 라이브러리가 설치되지 않았습니다. pip install matplotlib 명령어로 설치해주세요.")
    sys.exit(1)

# --- 전역 설정 ---
MOTOR_COM_PORT = 'COM5'
BAUD_RATE = 115200
STEPS_PER_MM = 12800
MOVE_DELAY_US = 500
PRELOAD_Z_MM = 1

# --- Helper Functions ---
def enqueue_output(process, queue):
    for line in iter(process.stdout.readline, ''):
        queue.put(line)
    process.stdout.close()

def log_stderr(process, name):
    for line in iter(process.stderr.readline, ''):
        print(f"[{name.upper()} STDERR] {line.strip()}", file=sys.stderr)
    process.stderr.close()

# --- 데이터 기록 스레드 함수 ---
def data_recorder(stop_event, lock, queues, writer, headers, base_coords, measure_axis, latest_sensor_data):
    try:
        writer.writerow(headers)
        base_x, base_y, base_z = base_coords
        while not stop_event.is_set():
            loop_start_time = time.time()
            with lock:
                try:
                    while True:
                        latest_sensor_data["laser_mm"] = float(queues['laser'].get_nowait().strip().split(':')[1])
                except (Empty, IndexError, ValueError): pass
                try:
                    while True:
                        parts = [int(p) for p in queues['due'].get_nowait().strip().split(',') if p]
                        if len(parts) == 16: latest_sensor_data["due_s_values"] = parts
                except (Empty, ValueError): pass
                try:
                    while True:
                        line = queues['afd50'].get_nowait().strip()
                        if line.startswith("Fx:"):
                            parts = line.split(',')
                            latest_sensor_data["afd_fx"] = float(parts[0].split(':')[1])
                            latest_sensor_data["afd_fy"] = float(parts[1].split(':')[1])
                            latest_sensor_data["afd_fz"] = float(parts[2].split(':')[1])
                except (Empty, IndexError, ValueError): pass
                data_to_write = latest_sensor_data.copy()

            gx, gy, gz = base_x, base_y, base_z
            laser_mm = data_to_write['laser_mm']
            if measure_axis == 'z':
                gz += laser_mm
            else:
                gz += laser_mm
                calculated_dist = data_to_write.get('calculated_distance', 0.0)
                if measure_axis == 'x': gx += calculated_dist
                elif measure_axis == 'y': gy += calculated_dist
            
            row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], f"{gx:.2f}", f"{gy:.2f}", f"{gz:.2f}", f"{laser_mm:.2f}"]
            row.extend([f"{data_to_write[k]:.2f}" for k in ["afd_fx", "afd_fy", "afd_fz"]])
            row.extend(data_to_write["due_s_values"])
            writer.writerow(row)

            if (sleep_time := 0.01 - (time.time() - loop_start_time)) > 0: time.sleep(sleep_time)
    except Exception as e:
        print(f"[RECORDER THREAD ERROR] {e}", file=sys.stderr)

# --- 메인 제어 함수 ---
def main():
    procs, queues, motor_ser, recorder_thread = {}, {}, None, None
    stop_event = threading.Event()

    try:
        print("자식 프로세스들을 시작합니다...")
        py_executable = sys.executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        procs['laser'] = subprocess.Popen([py_executable, '-u', os.path.join(script_dir, 'laser_reader.py')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        procs['due'] = subprocess.Popen([py_executable, '-u', os.path.join(script_dir, 'due_reader.py')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        procs['afd50'] = subprocess.Popen([py_executable, '-u', os.path.join(script_dir, 'afd50_reader.py')], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        for name, proc in procs.items():
            queues[name] = Queue()
            threading.Thread(target=enqueue_output, args=(proc, queues[name]), daemon=True).start()
            threading.Thread(target=log_stderr, args=(proc, name), daemon=True).start()
        print("모터 컨트롤러 연결...")
        motor_ser = serial.Serial(MOTOR_COM_PORT, BAUD_RATE, timeout=0.5)
        time.sleep(2)
        print("모든 프로세스 및 장치 준비 완료.")

        print("--- 측정 설정값을 입력해주세요 ---")
        coord_input = input("1. 라벨과 시작 글로벌 좌표 (예: P1,10.5,10.5,0): ")
        parts = [p.strip() for p in coord_input.split(',')]
        pi, base_x, base_y, base_z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        measure_axis = input("2. 측정할 축 (x, y, z): ").lower()
        if measure_axis not in ['x', 'y', 'z']: raise ValueError("측정 축은 x, y, z 중 하나여야 합니다.")
        
        target_abs_displacement = abs(float(input(f"3. {measure_axis.upper()}축 절대 이동 거리(mm): ")))
        test_count = int(input("4. 테스트 횟수(N)를 입력해주세요: "))
        print("---------------------------------")

        headers = ["Timestamp", "global_x", "global_y", "global_z", "laser_mm", "afd_fx", "afd_fy", "afd_fz"] + [f"s{j+1}" for j in range(16)]
        latest_sensor_data = {"laser_mm": 0.0, "due_s_values": [-1]*16, "afd_fx": 0.0, "afd_fy": 0.0, "afd_fz": 0.0, "calculated_distance": 0.0}
        lock = threading.Lock()
        increment_mm = 0.1

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        history_size = 200
        ax1.set_title("AFD-50 Force Data"); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Force (N)")
        lines_afd = [ax1.plot([], [], label=f'F{ax}')[0] for ax in ['x', 'y', 'z']]
        ax1.legend(); ax1.grid(True)
        ax2.set_title("DUE Sensor Data"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Sensor Value")
        linestyles = ['-', '--', ':', '-.']
        lines_due = [ax2.plot([], [], label=f's{i+1}', linestyle=linestyles[(i//10)%len(linestyles)])[0] for i in range(16)]
        ax2.grid(True); fig.tight_layout()
        plot_data = {'time': deque(maxlen=history_size), 'afd': [deque(maxlen=history_size) for _ in range(3)], 'due': [deque(maxlen=history_size) for _ in range(16)]}

        def update_display_and_plot(loop_start_time, title, current_target):
            with lock:
                display_data = latest_sensor_data.copy()
            
            current_time = time.time() - loop_start_time
            plot_data['time'].append(current_time)
            plot_data['afd'][0].append(display_data['afd_fx']); plot_data['afd'][1].append(display_data['afd_fy']); plot_data['afd'][2].append(display_data['afd_fz'])
            for i, val in enumerate(display_data['due_s_values']): plot_data['due'][i].append(val)
            for i, line in enumerate(lines_afd): line.set_data(plot_data['time'], plot_data['afd'][i])
            for i, line in enumerate(lines_due): line.set_data(plot_data['time'], plot_data['due'][i])
            ax1.relim(); ax1.autoscale_view(); ax2.relim(); ax2.autoscale_view()
            fig.canvas.draw(); fig.canvas.flush_events()
            
            os.system('cls' if os.name == 'nt' else 'clear')
            print(title)
            
            gx, gy, gz = base_x, base_y, base_z
            laser_mm = display_data['laser_mm']
            if measure_axis == 'z':
                gz += laser_mm
            else:
                gz += laser_mm
                calculated_dist = display_data.get('calculated_distance', 0.0)
                if measure_axis == 'x': gx += calculated_dist
                elif measure_axis == 'y': gy += calculated_dist

            print(f"글로벌 좌표: X={gx:.2f}, Y={gy:.2f}, Z={gz:.2f} (mm)")
            print(f"측정 축: {measure_axis.upper()}, 목표 변위: {current_target:.2f} mm")
            print(f"Laser Z 변위: {display_data['laser_mm']:.2f} mm")
            print(f"AFD-50 Force: Fx={display_data['afd_fx']: >7.2f}, Fy={display_data['afd_fy']: >7.2f}, Fz={display_data['afd_fz']: >7.2f} (N)")
            print("DUE Sensor (s1-s16):")
            s_vals = display_data['due_s_values']
            for i in range(0, 16, 4): print(f"  s{i+1:02d}-s{i+4:02d}: {s_vals[i]: >5}, {s_vals[i+1]: >5}, {s_vals[i+2]: >5}, {s_vals[i+3]: >5}")
            print("-" * 40)
            return 0 in display_data["due_s_values"]

        if measure_axis in ['x', 'y']:
            print(f"\n--- Z축 사전 접촉 이동 시작 (목표: {PRELOAD_Z_MM}mm) ---")
            z_steps = int(increment_mm * STEPS_PER_MM)
            while abs(latest_sensor_data['laser_mm']) < PRELOAD_Z_MM:
                motor_ser.write(f"M,Z,{z_steps},{MOVE_DELAY_US}\n".encode('utf-8'))
                time.sleep(0.7)
                try:
                    while True:
                        line = queues['laser'].get_nowait().strip()
                        if line.startswith("Dist_mm:"):
                            with lock: latest_sensor_data["laser_mm"] = float(line.split(':')[1])
                except Empty: pass
                print(f"  Z축 현재 위치: {latest_sensor_data['laser_mm']:.2f} mm", end='\r')
            motor_ser.write(b'S\n')
            print(f"\nZ축 사전 접촉 완료. 현재 위치: {latest_sensor_data['laser_mm']:.2f} mm")

        for i in range(1, test_count + 1):
            for direction_sign in [1, -1]:
                current_target_displacement = target_abs_displacement * direction_sign
                direction_label = "pos" if direction_sign == 1 else "neg"
                title_prefix = f"--- 테스트 {i}/{test_count} ({direction_label}) ---"
                print(f"\n{title_prefix}")

                for key in plot_data: 
                    if isinstance(plot_data[key], list): [d.clear() for d in plot_data[key]]
                    else: plot_data[key].clear()
                plot_start_time = time.time()

                filename = f"data_{pi}_{measure_axis.upper()}_{target_abs_displacement}mm_{i}_{direction_label}.csv"
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    stop_event.clear()
                    recorder_thread = threading.Thread(target=data_recorder, args=(stop_event, lock, queues, writer, headers, (base_x, base_y, base_z), measure_axis, latest_sensor_data))
                    recorder_thread.start()
                    print(f"데이터 기록 시작: {filename}")

                    direction = 1 if current_target_displacement >= 0 else -1
                    steps = int(increment_mm * STEPS_PER_MM) * direction

                    if measure_axis == 'z':
                        while abs(latest_sensor_data['laser_mm']) < abs(current_target_displacement):
                            motor_ser.write(f"M,Z,{steps},{MOVE_DELAY_US}\n".encode('utf-8'))
                            time.sleep(0.7)
                            if update_display_and_plot(plot_start_time, f"{title_prefix} 진행 중", current_target_displacement): print("\n접촉 감지! 이동 중지."); break
                    else: # X, Y
                        calculated_pos = 0.0
                        with lock: latest_sensor_data['calculated_distance'] = 0.0
                        while calculated_pos < abs(current_target_displacement):
                            motor_ser.write(f"M,{measure_axis.upper()},{steps},{MOVE_DELAY_US}\n".encode('utf-8'))
                            time.sleep(0.7)
                            calculated_pos += increment_mm
                            with lock: latest_sensor_data['calculated_distance'] = calculated_pos * direction_sign
                            if update_display_and_plot(plot_start_time, f"{title_prefix} 진행 중", current_target_displacement): print("\n접촉 감지! 이동 중지."); break
                    motor_ser.write(b'S\n')
                    print("\n이동 완료. 원점 복귀 시작...")

                    return_steps = -steps
                    if measure_axis == 'z':
                        while abs(latest_sensor_data['laser_mm']) > 0.05:
                            motor_ser.write(f"M,Z,{return_steps},{MOVE_DELAY_US}\n".encode('utf-8'))
                            time.sleep(0.7)
                            update_display_and_plot(plot_start_time, f"{title_prefix} 원점 복귀 중", 0)
                    else: # X, Y
                        calculated_pos = abs(latest_sensor_data['calculated_distance'])
                        while calculated_pos > 0.01:
                            motor_ser.write(f"M,{measure_axis.upper()},{return_steps},{MOVE_DELAY_US}\n".encode('utf-8'))
                            time.sleep(0.7)
                            calculated_pos -= increment_mm
                            with lock: latest_sensor_data['calculated_distance'] = (calculated_pos if calculated_pos > 0 else 0) * direction_sign
                            update_display_and_plot(plot_start_time, f"{title_prefix} {measure_axis.upper()}축 원점 복귀 중", 0)
                    
                    motor_ser.write(b'S\n')
                    print("원점 복귀 완료.")
                    stop_event.set()
                    recorder_thread.join()

            if i < test_count: 
                print(f"\n다음 테스트 세트까지 5초간 대기합니다...")
                time.sleep(5)

    except (KeyboardInterrupt, ValueError, ConnectionError) as e:
        print(f"\n오류 또는 중단: {e}")
    finally:
        print("정리 및 종료 절차를 시작합니다...")
        stop_event.set()
        if recorder_thread and recorder_thread.is_alive(): recorder_thread.join()
        if motor_ser: motor_ser.close()
        for proc in procs.values(): proc.terminate(); proc.wait()
        if 'fig' in locals():
            print("그래프 창을 닫으면 프로그램이 완전히 종료됩니다.")
            plt.ioff(); plt.show()
        print("프로그램 종료 완료.")

if __name__ == "__main__":
    main()