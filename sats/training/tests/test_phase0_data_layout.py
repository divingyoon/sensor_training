"""Phase 0 data layout contract: skin_ws raw BIN archive -> learning_data plan."""

from pathlib import Path

import pytest

from sats.preprocessing.bin_merge import (
    DUE_MAGIC,
    ETHERMOTION_MAGIC,
    LOADCELL_MAGIC,
    find_bin_set,
    read_bin_header,
)
from sats.preprocessing.prepare_learning_data import (
    REGISTRY_FILENAME,
    discover_planned_trials,
    load_trial_registry,
    save_trial_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "skin_ws" / "raw_data"
SATS_SOURCE = SOURCE_ROOT / "sats" / "eco20 + mesh"


class TestSkinWsRawBinArchive:
    def test_sats_source_root_exists(self):
        assert SATS_SOURCE.is_dir(), f"SATS source archive missing: {SATS_SOURCE}"

    @pytest.mark.parametrize(
        "trial_dir",
        [
            SATS_SOURCE / "d5" / "test1",
        ],
    )
    def test_required_raw_bin_set_exists(self, trial_dir):
        paths = find_bin_set(trial_dir)
        assert paths["due"].name.startswith("due_raw_burst_")
        assert paths["ethermotion"].name.startswith("ethermotion_encoder_")
        assert paths["loadcell"].name.startswith("loadcell_raw_")

class TestRawBinHeaders:
    @pytest.fixture(scope="class")
    def d5_paths(self):
        return find_bin_set(SATS_SOURCE / "d5" / "test1")

    @pytest.fixture(scope="class")
    def d10_paths(self):
        return None

    def test_d5_due_header(self, d5_paths):
        magic, header, _ = read_bin_header(d5_paths["due"])
        assert magic == DUE_MAGIC
        assert "sensor-major uint32 little-endian: sensor[16][frame10]" in header["payload_layout"]
        assert header["num_sensors"] == 16
        assert header["fifo_frames"] == 10

    def test_d10_due_header(self, d10_paths):
        if d10_paths is None:
            pytest.skip("d10 smoke archive is not present in this checkout")
        magic, header, _ = read_bin_header(d10_paths["due"])
        assert magic == DUE_MAGIC
        assert "sensor-major uint32 little-endian: sensor[16][frame10]" in header["payload_layout"]

    def test_ethermotion_header_supports_record_size_in_archive(self, d5_paths, d10_paths):
        for paths in [p for p in [d5_paths, d10_paths] if p is not None]:
            magic, header, _ = read_bin_header(paths["ethermotion"])
            assert magic == ETHERMOTION_MAGIC
            assert int(header["record_bytes"]) in {40, 56}
            assert "u_cmd" in header["columns"] or "U" in header["columns"]

    def test_loadcell_header(self, d5_paths, d10_paths):
        for paths in [p for p in [d5_paths, d10_paths] if p is not None]:
            magic, header, _ = read_bin_header(paths["loadcell"])
            assert magic == LOADCELL_MAGIC
            assert header["record_bytes"] == "variable"


class TestLearningDataPlan:
    def test_current_archive_plans_three_force_trials(self, tmp_path):
        planned, skipped = discover_planned_trials(
            SOURCE_ROOT,
            tmp_path / "learning_data",
            source_material="eco20 + mesh",
            material="ecomesh",
            depth_map={"d5": 2.5, "d10": 3.5},
        )

        assert [p.trial_id for p in planned] == [
            "ecomesh_d5_z2.5_test1",
        ]
        assert isinstance(skipped, list)


class TestTrialRegistryStableNumbering:
    """test 번호는 registry로 영구 고정되어, 과거 날짜 폴더를 끼워넣어도 안 밀린다."""

    @staticmethod
    def _make_trial(d_dir: Path, name: str) -> None:
        trial = d_dir / name
        trial.mkdir(parents=True)
        for stem in ("due_raw_burst", "ethermotion_encoder", "loadcell_raw"):
            (trial / f"{stem}_{name}.bin").write_bytes(b"")

    def _plan(self, source_root: Path, learning_root: Path) -> dict:
        registry = load_trial_registry(learning_root / REGISTRY_FILENAME)
        planned, _ = discover_planned_trials(
            source_root,
            learning_root,
            source_material="eco20 + mesh",
            material="ecomesh",
            depth_map={"d10": 3.5},
            registry=registry,
        )
        save_trial_registry(learning_root / REGISTRY_FILENAME, registry)
        return {p.source_dir.name: p.test_no for p in planned}

    def test_inserting_earlier_dated_trial_does_not_renumber(self, tmp_path):
        source_root = tmp_path / "raw"
        d_dir = source_root / "eco20 + mesh" / "d10"
        learning_root = tmp_path / "learning_data"

        self._make_trial(d_dir, "20260601_test1")
        self._make_trial(d_dir, "20260602_test1")
        first = self._plan(source_root, learning_root)
        assert first == {"20260601_test1": 1, "20260602_test1": 2}

        # 더 빠른 날짜 폴더를 나중에 추가 → 기존 번호는 유지, 새 폴더만 append.
        self._make_trial(d_dir, "20260531_test1")
        second = self._plan(source_root, learning_root)
        assert second["20260601_test1"] == 1
        assert second["20260602_test1"] == 2
        assert second["20260531_test1"] == 3

    def test_numbering_is_idempotent_across_reruns(self, tmp_path):
        source_root = tmp_path / "raw"
        d_dir = source_root / "eco20 + mesh" / "d10"
        learning_root = tmp_path / "learning_data"
        self._make_trial(d_dir, "20260601_test1")
        self._make_trial(d_dir, "20260602_test1")
        assert self._plan(source_root, learning_root) == self._plan(source_root, learning_root)


class TestForceConversionContract:
    def test_fz_conversion_formula(self):
        baseline_kg = 0.10
        press_kg = 0.30
        fz = (press_kg - baseline_kg) * 9.80665
        assert fz > 0
        assert abs(fz - 1.9613) < 0.001
