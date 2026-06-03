"""Phase 0 검증: 표준 폴더 구조, CSV 포맷, loadcell baseline"""
import os
import pandas as pd
import pytest

RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../raw_data/ecomesh")
D5_DIR = os.path.join(RAW_ROOT, "d5/z_2.5mm/test1")
D10_DIR = os.path.join(RAW_ROOT, "d10/z_2.0mm/test1")

EXPECTED_ETHERM_COLS = ["elapsed_ns", "time_s", "X", "Y", "Z", "U"]
EXPECTED_LOADCELL_COLS = ["elapsed_ns", "time_s", "kg"]
EXPECTED_DUE_COLS = (
    ["elapsed_ns", "time_s", "burst_index", "frame_index"]
    + [f"Skin{i}" for i in range(1, 17)]
)


# ── 0.1 폴더 구조 ──────────────────────────────────────────────────────────
class TestFolderStructure:
    def test_d5_standard_dir_exists(self):
        assert os.path.isdir(D5_DIR), f"d5 표준 폴더 없음: {D5_DIR}"

    def test_d10_standard_dir_exists(self):
        assert os.path.isdir(D10_DIR), f"d10 표준 폴더 없음: {D10_DIR}"

    @pytest.mark.parametrize("fname", [
        "ethermotion_data.csv", "loadcell_data.csv", "due_data.csv", "afd50_data.csv",
    ])
    def test_d5_required_files(self, fname):
        assert os.path.isfile(os.path.join(D5_DIR, fname))

    @pytest.mark.parametrize("fname", [
        "ethermotion_data.csv", "loadcell_data.csv", "due_data.csv", "afd50_data.csv",
    ])
    def test_d10_required_files(self, fname):
        assert os.path.isfile(os.path.join(D10_DIR, fname))


# ── 0.2 CSV 헤더·스케일·단위 ───────────────────────────────────────────────
class TestCSVFormat:
    @pytest.fixture(scope="class")
    def d5_eth(self):
        return pd.read_csv(os.path.join(D5_DIR, "ethermotion_data.csv"))

    @pytest.fixture(scope="class")
    def d10_eth(self):
        return pd.read_csv(os.path.join(D10_DIR, "ethermotion_data.csv"))

    @pytest.fixture(scope="class")
    def d5_lc(self):
        return pd.read_csv(os.path.join(D5_DIR, "loadcell_data.csv"))

    @pytest.fixture(scope="class")
    def d10_lc(self):
        return pd.read_csv(os.path.join(D10_DIR, "loadcell_data.csv"))

    # 헤더
    def test_d5_ethermotion_columns(self, d5_eth):
        assert list(d5_eth.columns) == EXPECTED_ETHERM_COLS

    def test_d10_ethermotion_columns(self, d10_eth):
        assert list(d10_eth.columns) == EXPECTED_ETHERM_COLS

    def test_d5_loadcell_columns(self, d5_lc):
        assert list(d5_lc.columns) == EXPECTED_LOADCELL_COLS

    def test_d10_loadcell_columns(self, d10_lc):
        assert list(d10_lc.columns) == EXPECTED_LOADCELL_COLS

    def test_d5_due_columns(self):
        df = pd.read_csv(os.path.join(D5_DIR, "due_data.csv"))
        assert list(df.columns) == EXPECTED_DUE_COLS

    def test_d10_due_columns(self):
        df = pd.read_csv(os.path.join(D10_DIR, "due_data.csv"))
        assert list(df.columns) == EXPECTED_DUE_COLS

    # afd50 데이터 없음(헤더만)
    def test_d5_afd50_empty(self):
        df = pd.read_csv(os.path.join(D5_DIR, "afd50_data.csv"))
        assert len(df) == 0, "afd50_data.csv는 헤더만 있어야 함(0행)"

    def test_d10_afd50_empty(self):
        df = pd.read_csv(os.path.join(D10_DIR, "afd50_data.csv"))
        assert len(df) == 0, "afd50_data.csv는 헤더만 있어야 함(0행)"

    # X/Y 스케일: ×1e-4 = ±10.0 mm
    def test_d5_xy_range(self, d5_eth):
        x_mm = d5_eth.X * 1e-4
        y_mm = d5_eth.Y * 1e-4
        assert x_mm.min() >= -10.01 and x_mm.max() <= 10.01
        assert y_mm.min() >= -10.01 and y_mm.max() <= 10.01

    def test_d10_xy_range(self, d10_eth):
        x_mm = d10_eth.X * 1e-4
        y_mm = d10_eth.Y * 1e-4
        assert x_mm.min() >= -10.01 and x_mm.max() <= 10.01
        assert y_mm.min() >= -10.01 and y_mm.max() <= 10.01

    # Z 범위: d5 press=15.5mm, d10 press=14.1mm
    def test_d5_z_press_max(self, d5_eth):
        z_max_mm = (d5_eth.Z * 1e-4).max()
        assert abs(z_max_mm - 15.5) < 0.05, f"d5 press Z 예상 15.5mm, 실제 {z_max_mm:.3f}mm"

    def test_d10_z_press_max(self, d10_eth):
        z_max_mm = (d10_eth.Z * 1e-4).max()
        assert abs(z_max_mm - 14.1) < 0.05, f"d10 press Z 예상 14.1mm, 실제 {z_max_mm:.3f}mm"

    # U 범위: 0~2.0mm (×1e-4)
    def test_d5_u_range(self, d5_eth):
        u_mm = d5_eth.U * 1e-4
        assert u_mm.min() >= 0.0
        assert u_mm.max() <= 2.01

    def test_d10_u_range(self, d10_eth):
        u_mm = d10_eth.U * 1e-4
        assert u_mm.min() >= 0.0
        assert u_mm.max() <= 2.01


# ── 0.3 loadcell baseline ─────────────────────────────────────────────────
class TestLoadcellBaseline:
    """
    baseline = 무부하 head 구간 kg 평균.
    스캔 시작 전(Z≈idle) 구간의 kg 평균이 합리적 범위에 있는지 확인.
    실제 baseline 자동추정 로직은 Phase 1에서 구현. 여기서는 데이터로 가능성만 검증.
    """

    def test_d5_loadcell_idle_kg_reasonable(self):
        df = pd.read_csv(os.path.join(D5_DIR, "loadcell_data.csv"))
        # 처음 1000행 = 스캔 시작 전 무부하 head 구간
        head_kg = df.kg.iloc[:1000].mean()
        assert 0.05 <= head_kg <= 0.20, f"d5 idle baseline 범위 이상: {head_kg:.4f} kg"

    def test_d10_loadcell_idle_kg_reasonable(self):
        df = pd.read_csv(os.path.join(D10_DIR, "loadcell_data.csv"))
        head_kg = df.kg.iloc[:1000].mean()
        assert 0.05 <= head_kg <= 0.20, f"d10 idle baseline 범위 이상: {head_kg:.4f} kg"

    def test_d5_loadcell_peak_physically_plausible(self):
        """압입 시 최대 force가 idle 대비 유의미하게 큼"""
        df = pd.read_csv(os.path.join(D5_DIR, "loadcell_data.csv"))
        baseline = df.kg.iloc[: max(1, len(df) // 20)].mean()
        peak = df.kg.max()
        assert peak > baseline + 0.1, f"d5 peak ({peak:.3f}kg) 가 baseline ({baseline:.3f}kg) 보다 충분히 크지 않음"

    def test_d10_loadcell_peak_physically_plausible(self):
        df = pd.read_csv(os.path.join(D10_DIR, "loadcell_data.csv"))
        baseline = df.kg.iloc[: max(1, len(df) // 20)].mean()
        peak = df.kg.max()
        assert peak > baseline + 0.1, f"d10 peak ({peak:.3f}kg) 가 baseline ({baseline:.3f}kg) 보다 충분히 크지 않음"

    def test_fz_conversion_formula(self):
        """N = (kg - baseline) × 9.80665 검증: 양수 압력 → 양수 Fz"""
        baseline_kg = 0.10
        press_kg = 0.30
        fz = (press_kg - baseline_kg) * 9.80665
        assert fz > 0, "압입 Fz는 양수여야 함"
        assert abs(fz - 1.9613) < 0.001
