import subprocess
import threading
import time
import sys
import os
from queue import Queue, Empty
from collections import deque
import numpy as np
import pandas as pd

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("pyqtgraph 라이브러리가 설치되지 않았습니다. pip install pyqtgraph PyQt6 명령어로 설치해주세요.")
    sys.exit(1)

# --- 상수 정의 ---
HISTORY_SIZE = 300       # 그래프에 표시할 데이터 포인트 수
UPDATE_FREQUENCY = 100   # Hz, 그래프 업데이트 주기
BASELINE_SAMPLES = 100   # Baseline 계산에 사용할 샘플 수
NOISE_THRESHOLD_N = 0.3  # 이 값(N) 이하의 힘은 노이즈로 간주하여 0으로 처리
CROSSTALK_ACTIVATION_THRESHOLD = 50 # 한 포인트의 센서들 평균 변화량이 이 값 이상이어야 유효 터치로 간주 (튜닝 필요)

# --- 센서-포인트 매핑 (0-based index) ---
POINT_SENSOR_MAPPING = {
    1: [0, 1, 4, 5], 3: [2, 3, 6, 7],
    7: [8, 9, 12, 13], 9: [10, 11, 14, 15]
}

# --- 헬퍼 함수 ---
def enqueue_output(process, queue):
    for line in iter(process.stdout.readline, ''):
        queue.put(line)
    process.stdout.close()

def log_stderr(process, name):
    for line in iter(process.stderr.readline, ''):
        print(f"[{name.upper()} STDERR] {line.strip()}", file=sys.stderr)
    process.stderr.close()

# --- 키 이벤트 처리를 위한 커스텀 위젯 ---
class CustomGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    key_pressed_signal = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.key_pressed_signal.emit(event)

# --- 메인 플로터 클래스 ---
class RealtimeForcePlotter:
    def __init__(self):
        self.procs = {}
        self.queues = {}
        self.c_matrix = None
        self.baseline = None
        self.start_time = time.time()
        self.is_recalibrating = threading.Event()

        self.point_ids = sorted(list(POINT_SENSOR_MAPPING.keys()))
        self.plot_data = {
            'time': deque(maxlen=HISTORY_SIZE),
            'forces': {p_i: [deque(maxlen=HISTORY_SIZE) for _ in range(3)] for p_i in self.point_ids}
        }
        self.app = None
        self.win = None
        self.plots = {}
        self.curves = {}

    def load_calibration_matrix(self):
        try:
            cal_files = [f for f in os.listdir('.') if f.startswith('cal_') and f.endswith('.csv')]
            if not cal_files: raise FileNotFoundError
            latest_cal_file = max(cal_files, key=os.path.getmtime)
            print(f"교정 파일 '{latest_cal_file}'을 로드합니다.")
            self.c_matrix = pd.read_csv(latest_cal_file, skiprows=1, nrows=3, header=0, index_col=0).values
            if self.c_matrix.shape != (3, 16): raise ValueError(f"C 행렬 차원 오류: {self.c_matrix.shape}")
            return True
        except FileNotFoundError:
            print("오류: 교정 파일(cal_*.csv)을 찾을 수 없습니다. calibration_cal.py를 먼저 실행하세요.")
        except Exception as e:
            print(f"오류: 교정 파일을 읽는 중 문제가 발생했습니다: {e}")
        return False

    def start_due_reader(self):
        print("DUE 리더 자식 프로세스를 시작합니다...")
        py_executable = sys.executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        due_script_path = os.path.join(script_dir, '..', 'acquisition_code', 'due_reader.py')
        if not os.path.exists(due_script_path):
            due_script_path = os.path.join(script_dir, 'acquisition_code', 'due_reader.py')
        if not os.path.exists(due_script_path):
            print(f"오류: due_reader.py를 찾을 수 없습니다: {due_script_path}")
            return False
        
        self.procs['due'] = subprocess.Popen(
            [py_executable, '-u', due_script_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore'
        )
        self.queues['due'] = Queue()
        threading.Thread(target=enqueue_output, args=(self.procs['due'], self.queues['due']), daemon=True).start()
        threading.Thread(target=log_stderr, args=(self.procs['due'], 'due'), daemon=True).start()
        print("프로세스 시작 완료. Baseline 측정을 시작합니다...")
        time.sleep(1)
        return True

    def _collect_baseline_samples(self):
        baseline_readings = []
        print(f"\n{BASELINE_SAMPLES}개의 샘플을 수집하여 Baseline을 재계산합니다...")
        
        while not self.queues['due'].empty():
            try: self.queues['due'].get_nowait()
            except Empty: break

        while len(baseline_readings) < BASELINE_SAMPLES:
            try:
                line = self.queues['due'].get(timeout=2).strip()
                if line:
                    try:
                        parts = [int(p) for p in line.split(',') if p]
                        if len(parts) == 16:
                            baseline_readings.append(parts)
                            print(f"  샘플 수집 중... {len(baseline_readings)}/{BASELINE_SAMPLES}", end='\r')
                    except ValueError:
                        pass
            except Empty:
                print("\n오류: Baseline 측정 중 DUE 리더로부터 데이터를 수신하지 못했습니다.")
                return None
        return baseline_readings

    def calculate_initial_baseline(self):
        readings = self._collect_baseline_samples()
        if readings is None: return False
        self.baseline = np.mean(readings, axis=0)
        print(f"\n초기 Baseline 계산 완료.")
        return True

    def re_calculate_baseline(self):
        if self.is_recalibrating.is_set():
            print("\n이미 Baseline 재계산이 진행 중입니다.")
            return

        def task():
            self.is_recalibrating.set()
            readings = self._collect_baseline_samples()
            if readings is not None:
                self.baseline = np.mean(readings, axis=0)
                print(f"\nBaseline 재계산 완료.")
            self.is_recalibrating.clear()

        threading.Thread(target=task, daemon=True).start()

    def handle_key_press(self, event):
        if event.key() == QtCore.Qt.Key.Key_R:
            self.re_calculate_baseline()

    def setup_plot(self):
        self.app = pg.mkQApp("Real-time Force Visualizer")
        self.win = CustomGraphicsLayoutWidget(show=True, title="실시간 3축 힘 센싱 (P1, P3, P7, P9) | 'r' 키로 영점 재설정")
        self.win.resize(1200, 800)
        self.win.ci.setBorder((50, 50, 100))
        self.win.key_pressed_signal.connect(self.handle_key_press)

        colors = ['r', 'g', 'b']
        labels = ['Fx', 'Fy', 'Fz']

        for i, p_i in enumerate(self.point_ids):
            row, col = divmod(i, 2)
            plot = self.win.addPlot(row=row, col=col, title=f"P{p_i}")
            plot.addLegend(size=(100,60), offset=(1, 1))
            plot.showGrid(x=True, y=True, alpha=0.3)
            self.plots[p_i] = plot
            self.curves[p_i] = {}
            for j in range(3):
                self.curves[p_i][j] = plot.plot(pen=colors[j], name=labels[j])

    def update(self):
        if self.is_recalibrating.is_set():
            return

        latest_due_values = None
        try:
            while True:
                line = self.queues['due'].get_nowait().strip()
                if line:
                    try:
                        parts = [int(p) for p in line.split(',') if p]
                        if len(parts) == 16:
                            latest_due_values = np.array(parts)
                    except ValueError:
                        pass
        except Empty:
            pass

        if latest_due_values is not None:
            current_time = time.time() - self.start_time
            self.plot_data['time'].append(current_time)

            s_corrected = latest_due_values - self.baseline
            
            for p_i in self.point_ids:
                active_sensors = POINT_SENSOR_MAPPING[p_i]
                
                mean_abs_change = np.mean(np.abs(s_corrected[active_sensors]))

                force_pi = np.zeros(3)
                if mean_abs_change >= CROSSTALK_ACTIVATION_THRESHOLD:
                    s_masked = np.zeros(16)
                    s_masked[active_sensors] = s_corrected[active_sensors]
                    force_pi = self.c_matrix @ s_masked
                
                force_pi[np.abs(force_pi) < NOISE_THRESHOLD_N] = 0
                
                for j in range(3):
                    self.plot_data['forces'][p_i][j].append(force_pi[j])
            
            time_data = list(self.plot_data['time'])
            for p_i in self.point_ids:
                for j in range(3):
                    self.curves[p_i][j].setData(time_data, list(self.plot_data['forces'][p_i][j]))

    def run(self):
        if not self.load_calibration_matrix(): return
        if not self.start_due_reader(): return
        if not self.calculate_initial_baseline(): return
        
        self.setup_plot()

        print("\n그래프가 시작되었습니다. 그래프 창이 활성화된 상태에서 'r' 키를 누르면 영점을 재설정합니다.")
        print("그래프 창을 닫거나 Ctrl+C를 누르면 종료됩니다.")
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(int(1000 / UPDATE_FREQUENCY))

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec()
        
        self.cleanup()

    def cleanup(self):
        print("\n정리 및 종료 절차를 시작합니다...")
        if 'due' in self.procs and self.procs['due'].poll() is None:
            self.procs['due'].terminate()
            self.procs['due'].wait()
        print("프로그램 종료 완료.")

if __name__ == '__main__':
    plotter = RealtimeForcePlotter()
    try:
        plotter.run()
    except Exception as e:
        print(f"\n치명적인 오류 발생: {e}")
        plotter.cleanup()
