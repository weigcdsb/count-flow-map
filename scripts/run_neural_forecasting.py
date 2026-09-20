"""Run the same formal neural experiment as notebooks/3_neural_forecasting.ipynb."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/neural_forecasting_v2.json')
    parser.add_argument('--device', default=None, help='cuda, cuda:0, or cpu; default auto-detect')
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'report'], default='all')
    parser.add_argument('--sessions', nargs='+', type=int, help='Run a subset of configured sessions; reports still require all sessions')
    parser.add_argument('--seeds', nargs='+', type=int, help='Run a subset of configured training seeds')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--download', action='store_true', help='Download the six configured sessions at the pinned revision first')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    import torch
    from countflow.neural_experiment import download_data, read_config, run_experiment
    from countflow.neural_training import TrainingPaused
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    config = read_config(args.config)
    if args.download or args.download_only:
        print(f'Downloading to {ROOT / config["data_root"]}', flush=True)
        download_data(config, ROOT)
    if args.download_only:
        return
    try:
        output = run_experiment(args.config, root=ROOT, device=args.device, stage=args.stage,
                                sessions=args.sessions, seeds=args.seeds)
    except (TrainingPaused, KeyboardInterrupt) as error:
        print(f'Paused. Rerun the same command to resume. {error}', flush=True)
        raise SystemExit(130)
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
