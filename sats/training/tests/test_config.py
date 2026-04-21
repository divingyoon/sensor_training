"""
tests/test_config.py

SATSConfig — attn_dim / lstm_ckpt 필드 TDD
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
