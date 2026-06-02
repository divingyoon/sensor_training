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
    from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
    pg.setConfigOptions(background='w', foreground='k')
except ImportError:
    print("pyqtgraph, PyQt6 라이브러리가 설치되지 않았습니다. pip install pyqtgraph PyQt6 명령어로 설치해주세요.")
    sys.exit(1)

# --- 상수 정의 ---
# 공간 좌표 및 포인트 정의 (실제 배치 좌표)
POINT_COORDINATES = {
    1: (3.25, 3.25), 3: (16.25, 3.25),
    7: (3.25, 16.25), 9: (16.25, 16.25)
}
HEATMAP_SIZE = (19.5, 19.5)  # cm 단위 정의(원본 좌표계)
UNIT_SCALE = 10.0            # cm -> mm 변환 배율
HEATMAP_SIZE_MM = (HEATMAP_SIZE[0] * UNIT_SCALE, HEATMAP_SIZE[1] * UNIT_SCALE)
HEATMAP_RESOLUTION_MM = 5.0  # mm 격자 해상도
HEATMAP_SIGMA_MM = 25.0      # mm 가우시안 폭 (기본 2.5cm)

# 시각화 감도 조절용 파라미터
# - Fz는 점의 크기/색상으로 반영
# - Fx, Fy는 화살표 길이/방향으로 반영
FZ_REF_N = 10.0              # Fz 기준 상한(N) — 색상/크기 정규화 기준
SCATTER_BASE_SIZE = 30        # 기본 점 크기
SCATTER_SIZE_PER_N = 10     # Fz 1N 당 점 크기 증가량
XY_VECTOR_SCALE = 0.15      # Fx, Fy 화살표 길이 스케일 (cm/뉴턴)
XY_VECTOR_SCALE_MM = XY_VECTOR_SCALE * UNIT_SCALE  # mm/뉴턴 스케일

# 화살표/선/텍스트 스타일 상수
ARROW_COLOR = (200, 200, 200, 200)  # RGBA
LINE_COLOR = (200, 200, 200, 200)    # RGBA
LINE_WIDTH = 5
ARROW_HEAD_LEN = 15
ARROW_TAIL_WIDTH = 2

TEXT_FONT_FAMILY = "Malgun Gothic"  # 또는 "Arial", "Noto Sans" 등
TEXT_FONT_SIZE_PT = 10
TEXT_FONT_BOLD = False

# 데이터 수집/처리 관련 상수
UPDATE_FREQUENCY = 100
BASELINE_SAMPLES = 100

# 활성화 임계값 관련 상수
PER_POINT_THRESHOLDS = {
    1: (20, 0.1), 3: (20 ,0.1),
    7: (20, 0.1), 9: (20, 0.1),
}
MIN_INDIVIDUAL_SENSOR_THRESHOLD = 15

# 글리프(Glyph) 시각화 관련 상수
GLYPH_RESOLUTION = 100 # (구) 글리프 해상도 — 사용하지 않지만 호환 유지
GAUSSIAN_SIGMA = 15    # (구) 글리프 분포 — 사용하지 않지만 호환 유지

# 센서-포인트 매핑
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

# --- 커스텀 위젯 ---
class CustomGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
    key_pressed_signal = QtCore.Signal(object)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        self.key_pressed_signal.emit(event)

# --- 메인 플로터 클래스 ---
class ForceGlyphPlotter:
    def __init__(self):
        self.procs = {}
        self.queues = {}
        self.c_matrix = None
        self.baseline = None
        self.is_recalibrating = threading.Event()
        
        self.point_ids = sorted(list(POINT_COORDINATES.keys()))
        self.forces = {p_i: np.zeros(3) for p_i in self.point_ids}

        # 시각화용 구성 요소
        self.app = None
        self.win = None
        self.plot = None
        self.heatmap_img = None
        self.scatter = None
        self.arrow_items = {}
        self.text_items = {}
        self.cmap = pg.colormap.get('magma') #'plasma', 'turbo', 'magma',   'cividis', 'gray'

        # 포인트 좌표 배열 (인덱스 정렬 보장)
        # 원본 좌표(cm) -> mm로 변환하여 사용
        point_pos_cm = np.array([POINT_COORDINATES[i] for i in self.point_ids], dtype=float)
        self.point_pos = point_pos_cm * UNIT_SCALE
        # 인접 포인트 간 최소 간격 계산(크로스토크 방지용 최대 화살표 길이)
        xs = np.unique(self.point_pos[:, 0])
        ys = np.unique(self.point_pos[:, 1])
        dx = float(np.min(np.diff(np.sort(xs)))) if xs.size > 1 else HEATMAP_SIZE_MM[0]
        dy = float(np.min(np.diff(np.sort(ys)))) if ys.size > 1 else HEATMAP_SIZE_MM[1]
        self.max_arrow_len = 0.45 * min(dx, dy)
        
        # 벡터 선(shaft) 아이템 저장소
        self.line_items = {}

        # 히트맵 격자 준비 (mm)
        gx = np.linspace(0, HEATMAP_SIZE_MM[0], int(HEATMAP_SIZE_MM[0] / HEATMAP_RESOLUTION_MM) + 1, dtype=np.float32)
        gy = np.linspace(0, HEATMAP_SIZE_MM[1], int(HEATMAP_SIZE_MM[1] / HEATMAP_RESOLUTION_MM) + 1, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(gx, gy)

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
            print("\n영점 재설정을 시작합니다...")
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
        self.app = pg.mkQApp("Force Field Viewer")
        self.win = CustomGraphicsLayoutWidget(show=True, title="실시간 힘 벡터 분포 | 'r' 키로 영점 재설정")
        self.win.resize(900, 900)
        self.win.key_pressed_signal.connect(self.handle_key_press)

        # 단일 플롯 위에 실제 좌표계(0..HEATMAP_SIZE)로 표현
        self.plot = self.win.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        self.plot.setRange(xRange=(0, HEATMAP_SIZE_MM[0]), yRange=(0, HEATMAP_SIZE_MM[1]))
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('bottom', 'X (mm)')
        self.plot.setLabel('left', 'Y (mm)')

        # Fz 히트맵 (배경)
        self.heatmap_img = pg.ImageItem()
        self.heatmap_img.setLookupTable(self.cmap.getLookupTable(nPts=256, alpha=True))
 
        # ImageItem은 setRect 시 내부 이미지의 width/height가 필요하므로
        # 먼저 초기 이미지를 설정한 뒤 rect를 지정한다.
        initial_surface = np.zeros_like(self.grid_x, dtype=np.float32)
        self.heatmap_img.setImage(initial_surface.T, autoLevels=False, levels=(0.0, FZ_REF_N))
        self.heatmap_img.setRect(QtCore.QRectF(0, 0, HEATMAP_SIZE_MM[0], HEATMAP_SIZE_MM[1]))
        self.heatmap_img.setZValue(-100)
        self.plot.addItem(self.heatmap_img)

        # 초기 점(포인트) - 크기/색상은 추후 업데이트에서 설정
        px, py = self.point_pos[:, 0], self.point_pos[:, 1]
        self.scatter = pg.ScatterPlotItem(px, py, size=SCATTER_BASE_SIZE, pen=pg.mkPen(70, 70, 70, 120), brush=pg.mkBrush(180, 180, 180, 150))
        self.plot.addItem(self.scatter)

        # 각 포인트에 대한 선분(shaft) + 화살표(head) + 텍스트(Fx,Fy,Fz) 준비
        for p_i, (px_i, py_i) in zip(self.point_ids, self.point_pos):
            # 선분(데이터 좌표 단위)
            line = pg.PlotCurveItem(pen=pg.mkPen(color=LINE_COLOR, width=LINE_WIDTH))
            self.plot.addItem(line)
            self.line_items[p_i] = line

            # 화살표 머리(픽셀 단위 사이즈)
            arrow = pg.ArrowItem(
                angle=0,
                headLen=ARROW_HEAD_LEN,
                tailLen=0,
                tailWidth=ARROW_TAIL_WIDTH,
                brush=pg.mkBrush(*ARROW_COLOR),
                pen=pg.mkPen(*ARROW_COLOR)
            )
            arrow.setPos(px_i, py_i)
            self.plot.addItem(arrow)
            self.arrow_items[p_i] = arrow

            txt = pg.TextItem(f"P{p_i}", anchor=(0.5, -0.2))
            font = QtGui.QFont(TEXT_FONT_FAMILY, TEXT_FONT_SIZE_PT)
            font.setBold(TEXT_FONT_BOLD)
            txt.setFont(font)
            txt.setPos(px_i, py_i)
            self.plot.addItem(txt)
            self.text_items[p_i] = txt

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
            for p_i in self.point_ids:
                activation_threshold, noise_threshold = PER_POINT_THRESHOLDS[p_i]
                active_sensors = POINT_SENSOR_MAPPING[p_i]
                s_point_changes = np.abs(s_corrected[active_sensors])
                
                if np.mean(s_point_changes) >= activation_threshold and np.all(s_point_changes >= MIN_INDIVIDUAL_SENSOR_THRESHOLD):
                    s_masked = np.zeros(16)
                    s_masked[active_sensors] = s_corrected[active_sensors]
                    force_vec = self.c_matrix @ s_masked
                    # 작은 노이즈 억제
                    force_vec[np.abs(force_vec) < noise_threshold] = 0.0
                    self.forces[p_i] = force_vec
                else:
                    self.forces[p_i] = np.zeros(3)

        # Fz 히트맵 갱신 (mm 좌표계)
        surface = np.zeros_like(self.grid_x, dtype=np.float32)
        for p_i, (px_i, py_i) in zip(self.point_ids, self.point_pos):
            fz_val = float(max(0.0, self.forces[p_i][2]))
            if fz_val <= 0.0:
                continue
            sq = (self.grid_x - px_i) ** 2 + (self.grid_y - py_i) ** 2
            surface += (fz_val * np.exp(-sq / (2.0 * HEATMAP_SIGMA_MM ** 2))).astype(np.float32)

        # 흰 배경 대비 레벨 고정 (0..FZ_REF_N)
        self.heatmap_img.setImage(surface.T, autoLevels=False, levels=(0.0, FZ_REF_N))

        # 시각 요소 업데이트: 점(색/크기) + 화살표 + 텍스트
        sizes = []
        brushes = []
        for idx, p_i in enumerate(self.point_ids):
            fx, fy, fz = self.forces[p_i]

            # 점 크기/색상: Fz 기반
            fz_clamped = max(0.0, fz)
            size = SCATTER_BASE_SIZE + SCATTER_SIZE_PER_N * min(FZ_REF_N, fz_clamped)
            sizes.append(size)

            norm = np.clip(fz_clamped / FZ_REF_N, 0.0, 1.0)
            rgba = self.cmap.map(norm)
            # cmap.map 반환 값 스케일 정규화 (0..1 또는 0..255 모두 허용)
            if np.max(rgba) <= 1.0:
                rgba = (rgba * 255).astype(np.uint8)
            color = pg.mkColor(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3]))
            brushes.append(pg.mkBrush(color))

            # 화살표(Fx,Fy) — 길이/방향
            arrow = self.arrow_items[p_i]
            vec_len = np.hypot(fx, fy)
            px_i, py_i = self.point_pos[idx]
            if vec_len > 1e-3:
                # 방향 단위벡터 및 목표 길이(스케일+최대 길이 제한)
                ux, uy = fx / vec_len, fy / vec_len
                target_len = min(vec_len * XY_VECTOR_SCALE_MM, self.max_arrow_len)
                dx = ux * target_len
                dy = uy * target_len

                # 선분(기점->끝점)을 데이터 좌표로 그림
                line = self.line_items[p_i]
                line.setData([px_i, px_i + dx], [py_i, py_i + dy])
                line.setVisible(True)

                # 화살표 머리(픽셀단위 사이즈), 데이터 좌표 위치
                angle = np.degrees(np.arctan2(dy, dx))
                arrow = self.arrow_items[p_i]
                arrow.setPos(px_i + dx, py_i + dy)
                arrow.setStyle(angle=angle, headLen=ARROW_HEAD_LEN, tailLen=0, tailWidth=ARROW_TAIL_WIDTH)
                arrow.setVisible(True)
            else:
                # 0에 가까운 벡터는 비가시화
                self.line_items[p_i].setVisible(False)
                self.arrow_items[p_i].setVisible(False)

            # 텍스트(Fx,Fy,Fz)
            self.text_items[p_i].setText(f"P{p_i}\nFx:{fx:.1f}  Fy:{fy:.1f}\nFz:{fz:.1f}")

        # 산점도 전체 업데이트 (크기/색상)
        self.scatter.setData(self.point_pos[:, 0], self.point_pos[:, 1], size=sizes, brush=brushes)

    def run(self):
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
    plotter = ForceGlyphPlotter()
    try:
        plotter.run()
    except Exception as e:
        print(f"\n치명적인 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        plotter.cleanup()
