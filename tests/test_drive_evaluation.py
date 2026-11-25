"""Real-data evaluation rejects malformed histories and incomplete contracts."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

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


def test_real_scene_images_can_use_an_overlay(tmp_path):
    root = tmp_path / 'raw'
    overlay = tmp_path / 'overlay'
    image = overlay / 'scene/frame.png'
    image.parent.mkdir(parents=True)
    Image.fromarray(np.full((2, 3, 3), 17, dtype=np.uint8)).save(image)
    row = {
        'messages': [{'content': [{'image': 'scene/frame.png'}] * 12 + [{'text': 'drive'}]}],
        'trajectory': {
            'hist_traj_10hz': [[0, 0, 0]] * 16,
            'hist_vel_10hz': [[0, 0]] * 16,
            'hist_acc_10hz': [[0, 0]] * 16,
            'ego_status': {},
            'nav_command': 2,
        },
    }
    observed = module().observation(row, root, overlay)
    assert sum(map(len, observed['views'].values())) == 12
    assert np.all(next(iter(observed['views'].values()))[0]['image'] == 17)
    assert observed['history'].shape == (16, 3)

    row['messages'][0]['content'][0] = {'image': '../outside.png'}
    with pytest.raises(ValueError, match='invalid scene image path'):
        module().observation(row, root, overlay)
