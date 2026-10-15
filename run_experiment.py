"""Print one paper training command; add --execute to actually train.

Run from the repository root. This launcher is not a cluster scheduler.
"""
import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LR_SEEDS = (13, 21, 42, 87, 100, 1234, 2024, 3407, 5678, 9999)
LRS = (1e-6, 5e-6, 1e-5, 3e-5, 4e-5, 5e-5)


def command(args):
    model = args.model
    is_pixel = model == 'pixel-base'
    task = 'mnli' if args.experiment in ('mnli-lr', 'mnli-longer') else args.task
    script = 'train_pixel_sweep.py' if is_pixel else 'train_subword_sweep.py'
    if args.experiment == 'mnli-lr':
        if args.seed not in LR_SEEDS or args.lr not in LRS:
            raise ValueError('Use a seed and learning rate from the recorded screen')
        if not is_pixel:
            script = 'train_bert_lr.py'
        options = ['--lr', str(args.lr), '--max-epochs', '80', '--screen-epochs', '10',
                   '--warmup-steps', '100', '--batch-size', '32', '--grad-accum', '4',
                   '--train-size', '24000', '--max-seq-length', '256', '--disable-early-stopping']
    elif args.experiment in ('sample-coverage', 'sst2-followup'):
        if args.experiment == 'sst2-followup':
            task = 'sst2'
            if not is_pixel or args.train_size != 24000 or args.seed not in (13,21,100,2024,3407,777,909):
                raise ValueError('The follow-up adds seven PIXEL SST-2 seeds at 24k only')
        elif args.seed not in (42, 1234, 5678):
            raise ValueError('Coverage comparison uses the three paired seeds')
        allowed = (2000, 8000, 24000) if task == 'sst2' else (8000, 24000)
        if args.train_size not in allowed:
            raise ValueError('Sample count was not part of this paper experiment')
        lr = 3e-5 if task == 'sst2' and is_pixel else 2e-5
        # The exact-update cap, not the epoch ceiling, sets the decay horizon.
        epochs = math.ceil(15000 / math.ceil(math.ceil(args.train_size / 32) / 4))
        options = ['--lr', str(lr), '--max-epochs', str(epochs), '--patience', '100000',
                   '--warmup-steps', '400', '--batch-size', '32', '--grad-accum', '4',
                   '--train-size', str(args.train_size), '--max-seq-length', '64' if task == 'sst2' else '256',
                   '--max-updates', '15000', '--eval-every-updates', '250' if task == 'sst2' else '500',
                   '--disable-early-stopping']
    else:
        if args.seed not in LR_SEEDS:
            raise ValueError('Longer-budget comparison uses the recorded ten seeds')
        options = ['--lr', '1e-5' if is_pixel else '2e-5',
                   '--max-epochs', '80' if is_pixel else '10',
                   '--patience', '12' if is_pixel else '3',
                   '--grad-accum', '4' if is_pixel else '1',
                   '--batch-size', '32', '--train-size', '24000', '--max-seq-length', '256']
        options += ['--warmup-steps', '400'] if is_pixel else ['--warmup-ratio', '0.1']
    suffix = f'_{args.experiment}_n{args.train_size}_s{args.seed}'
    if args.experiment == 'mnli-lr':
        suffix += f'_lr{args.lr:g}'
    output = Path('outputs/finetuning') / f'{task}_{model}{suffix}'
    if output.exists():
        raise FileExistsError('Run output already exists')
    return [sys.executable, str(ROOT / 'finetuning' / script),
            '--task', task, '--model', model, '--run-suffix', suffix,
            '--seed', str(args.seed)] + options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True,
                        choices=('mnli-lr','mnli-longer','sample-coverage','sst2-followup'))
    parser.add_argument('--model', required=True, choices=('bert','pixel-base'))
    parser.add_argument('--task', choices=('sst2','mnli'), default='sst2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--train-size', type=int, default=24000)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.execute:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
