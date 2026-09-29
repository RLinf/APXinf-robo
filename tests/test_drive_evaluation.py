"""Real-data evaluation rejects malformed histories and incomplete contracts."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('drive_evaluation', ROOT / 'scripts/eval_qwen_drive.py')
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def test_short_history_is_rejected_instead_of_interpolated(tmp_path):
    row = {'trajectory': {'hist_traj_10hz': [[0, 0, 0]] * 4}}
    with pytest.raises(ValueError, match='expected finite'):
        module().observation(row, tmp_path)


def test_score_only_requires_explicit_maps(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['eval_qwen_drive', '--score-only', '--scenes', 'scenes.jsonl',
                                     '--results-jsonl', 'predictions.jsonl', '--summary-json', 'summary.json',
                                     '--metric-cache', 'cache.csv'])
    with pytest.raises(SystemExit) as error:
        module().parse_args()
    assert error.value.code == 2


def test_score_only_needs_no_checkpoint_or_gpu(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['eval_qwen_drive', '--score-only', '--scenes', 'scenes.jsonl',
                                     '--results-jsonl', 'predictions.jsonl', '--summary-json', 'summary.json',
                                     '--metric-cache', 'cache.csv', '--maps', 'maps'])
    args = module().parse_args()
    assert args.score_only and args.model_dir is None and args.image_root is None
