#!/usr/bin/env python3
"""Evaluate Qwen-Drive on real NAVSIM scene records and official metric caches.

Scene JSONL uses Qwen-Drive's messages/trajectory/meta_info format. History must
contain 16 observed 10 Hz states; this evaluator never interpolates histories.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import lzma
import os
from pathlib import Path
import pickle
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'apxinf/python/apxinf'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--image-overlay', type=Path, help='fallback root for scene images missing under --image-root')
    p.add_argument('--tactics', type=Path)
    p.add_argument('--maps', type=Path, help='nuPlan maps root for PDM scoring')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--results-jsonl', type=Path, required=True)
    p.add_argument('--metric-cache', type=Path, help='NAVSIM metadata/cache.csv; only load trusted local caches')
    p.add_argument('--summary-json', type=Path, required=True)
    p.add_argument('--score-only', action='store_true', help='score existing predictions in a separate NAVSIM environment')
    args = p.parse_args()
    if args.metric_cache and not args.maps:
        p.error('--metric-cache requires --maps')
    if args.score_only and not args.metric_cache:
        p.error('--score-only requires --metric-cache')
    if not args.score_only and (not args.model_dir or not args.image_root):
        p.error('prediction requires --model-dir and --image-root')
    return args


def observation(row, root, overlay=None):
    from PIL import Image
    from apxinf.policies.impls.qwen_drive import CAMERA_VIEWS
    t = row['trajectory']
    history = {}
    for source, target, width in (('hist_traj_10hz', 'history', 3), ('hist_vel_10hz', 'history_velocity', 2), ('hist_acc_10hz', 'history_acceleration', 2)):
        value = np.asarray(t[source], dtype=np.float32)
        if value.shape != (16, width) or not np.isfinite(value).all():
            raise ValueError(f'{source}: expected finite [16,{width}], got {value.shape}')
        history[target] = value
    content = row['messages'][0]['content']
    frames = [item for item in content if 'image' in item]
    if len(frames) != 12:
        raise ValueError('evaluation requires three cameras with four frames each')
    decoded = []
    for item in frames:
        relative = Path(item['image'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'invalid scene image path: {relative}')
        image_path = root / relative
        if not image_path.is_file() and overlay is not None:
            image_path = overlay / relative
        with Image.open(image_path) as im:
            frame = {'image': np.array(im.convert('RGB'))}
        if 'resized_width' in item:
            frame['target_size'] = (item['resized_width'], item['resized_height'])
        decoded.append(frame)
    return dict(views={name: decoded[i*4:i*4+4] for i,name in enumerate(CAMERA_VIEWS)},
                **history, **t['ego_status'], nav_command=t['nav_command'],
                instruction_text=[item['text'] for item in content if 'text' in item][-1])


def main():
    a = parse_args()
    rows = [json.loads(line) for line in a.scenes.read_text().splitlines() if line.strip()]
    tokens = [r['meta_info']['token'] for r in rows]
    if not rows or len(tokens) != len(set(tokens)):
        raise ValueError('scene set must be nonempty with unique tokens')
    caches = {}
    if a.metric_cache:
        os.environ["NUPLAN_MAPS_ROOT"] = str(a.maps.resolve())
        os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
        for line in a.metric_cache.read_text().splitlines()[1:]:
            path = Path(line.strip())
            caches[path.parent.name] = path
        missing = set(tokens) - caches.keys()
        if missing:
            raise ValueError(f'missing metric caches for {len(missing)} scenes')
        from navsim.common.dataclasses import Trajectory, TrajectorySampling
        from navsim.evaluate.pdm_score import pdm_score
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        sampling = TrajectorySampling(num_poses=40, interval_length=.1)
        scorer, simulator = PDMScorer(sampling), PDMSimulator(sampling)
    if not a.score_only:
        from apxinf_robo import load_policy
        # Generate reference CUDA noise in a child process so Torch does not
        # preload its CUDA/cuBLAS libraries into the native inference process.
        code = """import json, sys, torch
noise = torch.randn((1,50,3), device='cuda', dtype=torch.float32,
                    generator=torch.Generator(device='cuda').manual_seed(int(sys.argv[1])))
print(json.dumps(noise.cpu().tolist()))
"""
        try:
            generated = subprocess.run([sys.executable, '-c', code, str(a.seed)],
                                       check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'CUDA PyTorch noise generation failed: {error.stderr}') from error
        noise = np.asarray(json.loads(generated.stdout), dtype=np.float32)
        if noise.shape != (1,50,3) or not np.isfinite(noise).all():
            raise ValueError('invalid reference initial noise')
        policy = load_policy(a.model_dir, precision='bf16', planner=a.model_dir/'planner-sft',
                             mode='direct_planning', num_steps=10, tactics=a.tactics)
        try:
            a.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with a.results_jsonl.open('x') as out:
                for row, token in zip(rows, tokens):
                    actions = np.asarray(policy.infer(observation(row, a.image_root, a.image_overlay), noise=noise)['actions']).copy()
                    if actions.shape != (50,3) or not np.isfinite(actions).all():
                        raise ValueError(f'{token}: invalid trajectory')
                    out.write(json.dumps(dict(token=token, trajectories=actions[None].tolist()))+'\n')
                    out.flush()
        finally:
            policy.close()
    predictions = [json.loads(line) for line in a.results_jsonl.read_text().splitlines() if line.strip()]
    if len(predictions) != len(tokens) or {r['token'] for r in predictions} != set(tokens):
        raise ValueError('prediction tokens do not match the complete scene set')
    scores = []
    components = []
    if caches:
        for prediction in predictions:
            actions = np.asarray(prediction['trajectories'], dtype=np.float32)
            if actions.shape != (1,50,3) or not np.isfinite(actions).all():
                raise ValueError('expected finite prediction [1,50,3]')
            with lzma.open(caches[prediction['token']], 'rb') as f:
                cache = pickle.load(f)
            metrics = {k:float(v) for k,v in asdict(pdm_score(cache, Trajectory(poses=actions[0,4:40:5]), sampling, simulator, scorer)).items()}
            if not np.isfinite(list(metrics.values())).all():
                raise ValueError('non-finite PDM components')
            scores.append(metrics['score'])
            components.append(dict(token=prediction['token'], **metrics))
        a.summary_json.parent.mkdir(parents=True, exist_ok=True)
        a.summary_json.with_suffix('.scenes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in components))
    summary = dict(predicted=len(tokens), scored=len(scores), seed=a.seed,
                   pdm=100*float(np.mean(scores)) if scores else None)
    a.summary_json.parent.mkdir(parents=True, exist_ok=True)
    a.summary_json.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
