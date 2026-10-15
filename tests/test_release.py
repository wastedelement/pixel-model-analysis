"""CPU checks for packaging and paper recipe consistency; no training."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiment


def release_files():
    excluded = {'.git', '__pycache__', 'outputs', 'data', 'checkpoints',
                'vendor', '.cache', '.venv', 'venv'}
    for path in ROOT.rglob('*'):
        if not path.is_file() or excluded.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix in ('.py', '.md', '.txt', '.json') or path.name == '.gitignore':
            yield path


class ReleaseTests(unittest.TestCase):
    def args(self, **updates):
        values = dict(experiment='mnli-lr', model='pixel-base', task='mnli',
                      seed=42, lr=3e-5, train_size=24000, execute=False)
        values.update(updates)
        return argparse.Namespace(**values)

    def test_all_python_sources_parse(self):
        for path in release_files():
            if path.suffix == '.py':
                ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))

    def test_lr_schedule_does_not_collapse_to_observation_window(self):
        for model in ('bert', 'pixel-base'):
            command = run_experiment.command(self.args(model=model))
            self.assertEqual(command[command.index('--max-epochs') + 1], '80')
            self.assertEqual(command[command.index('--screen-epochs') + 1], '10')
            self.assertEqual(command[command.index('--grad-accum') + 1], '4')
            self.assertIn('--disable-early-stopping', command)
            self.assertNotIn('--max-updates', command)

    def test_sample_budget_is_fixed(self):
        for task, amounts in [('sst2',(2000,8000,24000)), ('mnli',(8000,24000))]:
            for model in ('bert','pixel-base'):
                for n in amounts:
                    command = run_experiment.command(self.args(
                        experiment='sample-coverage', model=model, task=task, train_size=n))
                    self.assertEqual(command[command.index('--max-updates')+1], '15000')
                    self.assertEqual(command[command.index('--warmup-steps')+1], '400')

    def test_unreported_search_points_rejected(self):
        with self.assertRaises(ValueError):
            run_experiment.command(self.args(lr=3.5e-5))
        with self.assertRaises(ValueError):
            run_experiment.command(self.args(experiment='sample-coverage', train_size=16000))

    def test_recorded_counts_and_all_failures_retained(self):
        data = json.loads((ROOT/'results/training.json').read_text())
        self.assertEqual(len(data['mnli_lr_pixel']),60)
        self.assertEqual(len(data['mnli_lr_bert']),60)
        self.assertEqual(len(data['sample_coverage']),30)
        self.assertEqual(sum(r['success'] for r in data['mnli_lr_bert']),60)
        for rate in (1e-6,5e-5):
            self.assertEqual(sum(r['success'] for r in data['mnli_lr_pixel'] if r['lr']==rate),0)

    def test_no_private_paths_or_host_recording(self):
        for path in release_files():
            content = path.read_text()
            for forbidden in ('/'+'Users/', '/'+'public/', '/'+'home/',
                              'gethost'+'name(', 'platform.'+'node('):
                self.assertNotIn(forbidden, content, str(path.relative_to(ROOT)))


if __name__ == '__main__':
    unittest.main()
