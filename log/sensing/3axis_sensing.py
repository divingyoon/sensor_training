import subprocess
import threading
import time
import sys
import os
from queue import Queue, Empty
import numpy as np
import pandas as pd

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
except ImportError:
    print("pyqtgraph 라이브러리가 설치되지 않았습니다. pip install pyqtgraph PyQt6 명령어로 설치해주세요.")
    sys.exit(1)

# --- 상수 정의 ---
HISTORY_SIZE = 300
UPDATE_FREQUENCY = 100
BASELINE_SAMPLES = 500
NOISE_THRESHOLD_N = 0.3

# --- 포인트별 튜닝 파라미터 ---
# 각 포인트의 특성에 맞게 활성화 임계값과 노이즈 제거 임계값을 개별적으로 설정합니다.
# 형식: {포인트_번호: (활성화_임계값, 노이즈_제거_임계값)}
# P1의 깜빡임 현상을 줄이기 위해 활성화 임계값을 다른 포인트보다 높게 설정했습니다.
PER_POINT_THRESHOLDS = {
    1: (100, 0.3), 2: (50, 0.3), 3: (20, 0.3),
    4: (50, 0.3), 5: (30, 0.3), 6: (50, 0.3),
    7: (80, 0.3), 8: (80, 0.3), 9: (50, 0.3),
}
# 포인트가 활성화되기 위해 해당 포인트의 모든 센서가 넘어야 하는 최소 변화량
MIN_INDIVIDUAL_SENSOR_THRESHOLD = 20

# --- 히트맵 시각화 상수 ---
POINT_COORDINATES = {
    1: (3.25, 3.25), 2: (9.75, 3.25), 3: (16.25, 3.25),
    4: (3.25, 9.75), 5: (9.75, 9.75), 6: (16.25, 9.75),
    7: (3.25, 16.25), 8: (9.75, 16.25), 9: (16.25, 16.25)
}
HEATMAP_SIZE = (19.5, 19.5)
HEATMAP_RESOLUTION = 0.5
GAUSSIAN_SIGMA = 2.5

# --- 센서-포인트 매핑 ---
POINT_SENSOR_MAPPING = {
    1: [0, 1, 4, 5], 2: [1, 2, 5, 6], 3: [2, 3, 6, 7],
    4: [4, 5, 8, 9], 5: [5, 6, 9, 10], 6: [6, 7, 10, 11],
    7: [8, 9, 12, 13], 8: [9, 10, 13, 14], 9: [10, 11, 14, 15]
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

# --- 커스텀 위젯 ---
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
        self.is_recalibrating = threading.Event()
        self.forces = {p_i: np.zeros(3) for p_i in range(1, 10)}

        grid_dim = int(HEATMAP_SIZE[0] / HEATMAP_RESOLUTION)
        self.grid_shape = (grid_dim, grid_dim)
        x = np.linspace(0, HEATMAP_SIZE[0], grid_dim)
        y = np.linspace(0, HEATMAP_SIZE[1], grid_dim)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

        self.app = None
        self.win = None
        self.heatmap_item = None
        self.point_markers = None

    def load_calibration_matrix(self):
        try:
            cal_files = [f for f in os.listdir('.') if f.startswith('cal_') and f.endswith('.csv')]
            if not cal_files: raise FileNotFoundError
            latest_cal_file = max(cal_files, key=os.path.getmtime)
            print(f"교정 파일 '{latest_cal_file}'을 로드합니다.")
            self.c_matrix = pd.read_csv(latest_cal_file, skiprows=1, nrows=3, header=0, index_col=0).values
            return True
        except Exception as e:
            print(f"오류: 교정 파일을 읽는 중 문제가 발생했습니다: {e}")
        return False

    def start_due_reader(self):
        py_executable = sys.executable
        script_dir = os.path.dirname(os.path.abspath(__file__))
        due_script_path = os.path.join(script_dir, '..', 'acquisition_code', 'due_reader.py')
        if not os.path.exists(due_script_path):
            due_script_path = os.path.join(script_dir, 'acquisition_code', 'due_reader.py')
        if not os.path.exists(due_script_path):
            print(f"오류: due_reader.py를 찾을 수 없습니다: {due_script_path}")
            return False
        self.procs['due'] = subprocess.Popen([py_executable, '-u', due_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        self.queues['due'] = Queue()
        threading.Thread(target=enqueue_output, args=(self.procs['due'], self.queues['due']), daemon=True).start()
        threading.Thread(target=log_stderr, args=(self.procs['due'], 'due'), daemon=True).start()
        time.sleep(1)
        return True

    def _collect_baseline_samples(self):
        baseline_readings = []
        print(f"\n{BASELINE_SAMPLES}개의 샘플로 Baseline을 계산합니다...")
        while not self.queues['due'].empty():
            try: self.queues['due'].get_nowait()
            except Empty: break
        while len(baseline_readings) < BASELINE_SAMPLES:
            try:
                line = self.queues['due'].get(timeout=2).strip()
                if line and len(parts := [int(p) for p in line.split(',') if p]) == 16:
                    baseline_readings.append(parts)
                    print(f"  샘플 수집 중... {len(baseline_readings)}/{BASELINE_SAMPLES}", end='\r')
            except (Empty, ValueError):
                continue
        if not baseline_readings:
            print("\n오류: Baseline 측정 중 데이터를 수신하지 못했습니다.")
            return None
        return baseline_readings

    def calculate_initial_baseline(self):
        readings = self._collect_baseline_samples()
        if readings is None: return False
        self.baseline = np.mean(readings, axis=0)
        print(f"\n초기 Baseline 계산 완료.")
        return True

    def re_calculate_baseline(self):
        if self.is_recalibrating.is_set(): return
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
        self.win = CustomGraphicsLayoutWidget(show=True, title="실시간 압력 분포 | 'r' 키로 영점 재설정")
        self.win.resize(800, 800)
        self.win.ci.setBorder((50, 50, 100))
        self.win.key_pressed_signal.connect(self.handle_key_press)
        plot = self.win.addPlot(row=0, col=0)
        plot.setAspectLocked(True)
        plot.setLabel('bottom', "X (mm)")
        plot.setLabel('left', "Y (mm)")
        plot.setRange(xRange=(-1, HEATMAP_SIZE[0] + 1), yRange=(-1, HEATMAP_SIZE[1] + 1))
        self.heatmap_item = pg.ImageItem()
        # BUG FIX: Initialize image with blank data before adding to plot
        self.heatmap_item.setImage(np.zeros(self.grid_shape))
        self.heatmap_item.setRect(QtCore.QRectF(0, 0, HEATMAP_SIZE[0], HEATMAP_SIZE[1]))
        plot.addItem(self.heatmap_item)
        colormap = pg.colormap.get('viridis')
        colorbar = pg.ColorBarItem(values=(0, 5), colorMap=colormap)
        colorbar.setImageItem(self.heatmap_item)
        self.win.addItem(colorbar, row=0, col=1)
        marker_coords = list(POINT_COORDINATES.values())
        self.point_markers = pg.ScatterPlotItem(x=[c[0] for c in marker_coords], y=[c[1] for c in marker_coords], pen=pg.mkPen('r', width=2), brush=pg.mkBrush(None), size=10, symbol='+')
        plot.addItem(self.point_markers)

    def update(self):
        if self.is_recalibrating.is_set(): return
        latest_due_values = None
        try:
            while True:
                line = self.queues['due'].get_nowait().strip()
                if line and len(parts := [int(p) for p in line.split(',') if p]) == 16:
                    latest_due_values = np.array(parts)
        except (Empty, ValueError):
            pass
        if latest_due_values is not None:
            s_corrected = latest_due_values - self.baseline
            for p_i in range(1, 10):
                # 포인트별 튜닝된 임계값 가져오기
                activation_threshold, noise_threshold = PER_POINT_THRESHOLDS[p_i]
                active_sensors = POINT_SENSOR_MAPPING[p_i]
                s_point_changes = np.abs(s_corrected[active_sensors])
                
                # 활성화 조건 강화: 평균 변화량이 임계값을 넘고, 모든 개별 센서 변화량도 최소 임계값을 넘어야 함
                if np.mean(s_point_changes) >= activation_threshold and np.all(s_point_changes >= MIN_INDIVIDUAL_SENSOR_THRESHOLD):
                    s_masked = np.zeros(16)
                    s_masked[active_sensors] = s_corrected[active_sensors]
                    force_pi = self.c_matrix @ s_masked
                    force_pi[np.abs(force_pi) < noise_threshold] = 0
                    self.forces[p_i] = force_pi
                else:
                    self.forces[p_i] = np.zeros(3)
        else:
            self.forces = {p_i: np.zeros(3) for p_i in range(1, 10)}
        heatmap_data = np.zeros(self.grid_shape)
        for p_i, force in self.forces.items():
            fz = force[2]
            if fz > 0:
                px, py = POINT_COORDINATES[p_i]
                sq_dist = (self.grid_x - px)**2 + (self.grid_y - py)**2
                heatmap_data += fz * np.exp(-sq_dist / (2 * GAUSSIAN_SIGMA**2))
        self.heatmap_item.setImage(heatmap_data.T)

    def run(self):
        if os.path.exists("debug_log.txt"):
            try: os.remove("debug_log.txt")
            except OSError:
                pass
        print("Starting... ")
        if not self.load_calibration_matrix(): return
        if not self.start_due_reader(): return
        if not self.calculate_initial_baseline(): return
        self.setup_plot()
        print("\n그래프가 시작되었습니다. 그래프 창이 활성화된 상태에서 'r' 키를 누르면 영점을 재설정합니다.")
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
        import traceback
        traceback.print_exc()
        plotter.cleanup()
