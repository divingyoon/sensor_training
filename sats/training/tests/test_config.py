"""
tests/test_config.py

SATSConfig — attn_dim / lstm_ckpt / local_map 필드 TDD
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from sats.training.config import SATSConfig


class TestSATSConfigAttentionFields:
    """Phase 1: attn_dim, lstm_ckpt 신규 필드 검증."""

    def test_attn_dim_default(self):
        cfg = SATSConfig()
        assert cfg.attn_dim == 64, "기본 attn_dim은 64여야 한다"

    def test_attn_dim_custom(self):
        cfg = SATSConfig(attn_dim=128)
        assert cfg.attn_dim == 128

    def test_lstm_ckpt_default_empty(self):
        cfg = SATSConfig()
        assert cfg.lstm_ckpt == "", "lstm_ckpt 기본값은 빈 문자열이어야 한다"

    def test_lstm_ckpt_custom(self):
        cfg = SATSConfig(lstm_ckpt="sats/training/runs/lstm_v1/best_model.pt")
        assert cfg.lstm_ckpt == "sats/training/runs/lstm_v1/best_model.pt"

    def test_existing_fields_unchanged(self):
        """기존 필드가 그대로 유지되는지 회귀 검증."""
        cfg = SATSConfig()
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 2
        assert cfg.n_sensors == 16
        assert cfg.grid_size == 40
        assert cfg.bidirectional is False


class TestSATSConfigLocalMapFields:
    """Phase 1 (Local Map): local_map_size, sensor_spacing_mm, attn_ckpt 필드 검증."""

    def test_local_map_size_default(self):
        cfg = SATSConfig()
        assert cfg.local_map_size == 15, "기본 local_map_size는 15여야 한다"

    def test_local_map_size_custom(self):
        cfg = SATSConfig(local_map_size=11)
        assert cfg.local_map_size == 11

    def test_local_map_size_is_odd(self):
        """기본값이 홀수여야 local map 중앙 배치가 정확하다."""
        cfg = SATSConfig()
        assert cfg.local_map_size % 2 == 1, "local_map_size는 홀수여야 한다"

    def test_sensor_spacing_mm_default(self):
        cfg = SATSConfig()
        assert cfg.sensor_spacing_mm == 6.5, "기본 sensor_spacing_mm는 6.5여야 한다"

    def test_sensor_spacing_mm_custom(self):
        cfg = SATSConfig(sensor_spacing_mm=5.0)
        assert cfg.sensor_spacing_mm == 5.0

    def test_attn_ckpt_default_empty(self):
        cfg = SATSConfig()
        assert cfg.attn_ckpt == "", "attn_ckpt 기본값은 빈 문자열이어야 한다"

    def test_attn_ckpt_custom(self):
        cfg = SATSConfig(attn_ckpt="sats/training/runs/attn_v1/best_model.pt")
        assert cfg.attn_ckpt == "sats/training/runs/attn_v1/best_model.pt"

    def test_existing_fields_unchanged_after_local_map_addition(self):
        """Local Map 필드 추가 후 기존 필드 회귀 검증."""
        cfg = SATSConfig()
        assert cfg.attn_dim == 64
        assert cfg.lstm_ckpt == ""
        assert cfg.hidden_dim == 64
        assert cfg.grid_size == 40
        assert cfg.grid_step_mm == 0.5
        assert cfg.grid_min_mm == -9.75


class TestSATSConfigCNNFields:
    """Phase 1 (CNN): local_map_ckpt, cnn_hidden_channels 필드 검증."""

    def test_local_map_ckpt_default_empty(self):
        cfg = SATSConfig()
        assert cfg.local_map_ckpt == "", "local_map_ckpt 기본값은 빈 문자열이어야 한다"

    def test_local_map_ckpt_custom(self):
        cfg = SATSConfig(local_map_ckpt="sats/training/runs/local_map_v1/best_model.pt")
        assert cfg.local_map_ckpt == "sats/training/runs/local_map_v1/best_model.pt"

    def test_cnn_hidden_channels_default(self):
        cfg = SATSConfig()
        assert cfg.cnn_hidden_channels == 16, "기본 cnn_hidden_channels는 16이어야 한다"

    def test_cnn_hidden_channels_custom(self):
        cfg = SATSConfig(cnn_hidden_channels=32)
        assert cfg.cnn_hidden_channels == 32

    def test_existing_fields_unchanged_after_cnn_addition(self):
        """CNN 필드 추가 후 기존 필드 회귀 검증."""
        cfg = SATSConfig()
        assert cfg.attn_dim == 64
        assert cfg.lstm_ckpt == ""
        assert cfg.attn_ckpt == ""
        assert cfg.local_map_size == 15
        assert cfg.hidden_dim == 64
        assert cfg.grid_size == 40
