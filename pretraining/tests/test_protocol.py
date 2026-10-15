"""Dependency-light checks; GPU training is a separate verification step."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_utils
import protocol


class ProtocolTests(unittest.TestCase):
    def test_update_budgets(self):
        self.assertEqual(protocol.updates_per_epoch(), 188)
        self.assertEqual(protocol.updates_per_epoch() * protocol.SCREEN_EPOCHS, 1880)
        self.assertEqual(protocol.updates_per_epoch() * protocol.SCHEDULE_EPOCHS, 15040)

    def test_warmup_is_seed_specific(self):
        self.assertEqual(protocol.warmup_steps(100), 100)
        self.assertEqual(protocol.warmup_steps(5678), 0)
        with self.assertRaises(ValueError):
            protocol.warmup_steps(42)

    def test_criterion_boundaries(self):
        self.assertTrue(protocol.training_fit({"train_loss": .9, "train_acc": .51}))
        self.assertFalse(protocol.training_fit({"train_loss": 1., "train_acc": .9}))
        self.assertFalse(protocol.training_fit({"train_loss": .9, "train_acc": .5}))

    def test_sampler_is_fixed_across_training_seeds(self):
        rows = [{"text": str(i), "label": i % 2} for i in range(100)]
        sampled = data_utils.stratified_subsample(rows, 20)
        self.assertEqual(len(sampled), 20)
        self.assertEqual(sum(row["label"] for row in sampled), 10)
        self.assertEqual(sampled, data_utils.stratified_subsample(rows, 20))
        self.assertEqual(data_utils.stratified_subsample(rows, None), rows)

    def test_mnli_label_order(self):
        self.assertEqual(data_utils.MNLI_LABEL2ID, {"entailment": 0, "neutral": 1, "contradiction": 2})

    def test_mnli_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [{"sentence1": " A ", "sentence2": " B ", "gold_label": label}
                       for label in ("entailment", "neutral", "contradiction", "-")]
            Path(tmp, "multinli_1.0_train.jsonl").write_text("\n".join(map(json.dumps, records)))
            original = data_utils.MNLI_DIR
            try:
                data_utils.MNLI_DIR = tmp
                rows = data_utils.load_mnli("train")
            finally:
                data_utils.MNLI_DIR = original
            self.assertEqual([row["label"] for row in rows], [0, 1, 2])
            self.assertEqual(rows[0]["premise"], "A")

    def test_source_files_parse(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(), filename=path.name)

    def test_transfer_retains_positions_and_heads(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is not installed; tensor transfer was not exercised")
        import torch
        from mnli import transfer_encoder

        class Encoder:
            def __init__(self):
                self.state = {"embeddings.position_embeddings": torch.ones(1, 257, 4),
                              "layer.weight": torch.zeros(4, 4)}

            def state_dict(self):
                return dict(self.state)

            def load_state_dict(self, state, strict):
                self.state = dict(state)

        class Model:
            def __init__(self):
                self.vit = Encoder()
                self.head = torch.zeros(3, 4)

        model = Model()
        source = {"embeddings.position_embeddings": torch.zeros(1, 145, 4),
                  "layer.weight": torch.ones(4, 4)}
        position = model.vit.state["embeddings.position_embeddings"].clone()
        rng = torch.get_rng_state().clone()
        transfer_encoder(model, source, torch)
        self.assertTrue(torch.equal(position, model.vit.state["embeddings.position_embeddings"]))
        self.assertTrue(torch.equal(model.vit.state["layer.weight"], source["layer.weight"]))
        self.assertTrue(torch.equal(model.head, torch.zeros(3, 4)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
