"""Re-evaluate a saved SST-2 checkpoint; never modify its source run.

Use the adjacent source_snapshot containing the server training/data code.
Training metrics here are full-subset, eval-mode checkpoint measurements,
unlike the online epoch metrics in training_log.jsonl.
"""
import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_pixel_sweep import (
    PixelDataset, PangoCairoTextRenderer, PIXELForSequenceClassification,
    PoolingMode, get_transforms,
)
from train_subword_sweep import TextDataset
from data_utils import get_train, get_evals, stratified_subsample


def digest(items):
    return hashlib.sha256(json.dumps(items, ensure_ascii=False,
                                   sort_keys=True).encode()).hexdigest()


def measure(model, dataset, device):
    model.eval()
    labels, predictions, probabilities = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            count = batch["labels"].numel()
            total_loss += float(out.loss) * count
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(out.logits.argmax(-1).cpu().tolist())
            probabilities.extend(out.logits.softmax(-1).cpu().tolist())
    confusion = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1
    return {"n": len(labels), "accuracy": float(np.mean(np.array(labels) == predictions)),
            "loss": total_loss / len(labels), "confusion_true_rows": confusion,
            "predicted_class_counts": dict(Counter(predictions)),
            "labels": labels, "predictions": predictions, "probabilities": probabilities}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-verified", action="store_true")
    parser.add_argument("--allow-low-fit", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text())
    best = json.loads((args.run_dir / "ckpt/best_info.json").read_text())
    records = [json.loads(line) for line in (args.run_dir / "training_log.jsonl").read_text().splitlines()]
    assert config["task"] == "sst2" and config["train_n"] == 24000
    assert config["max_updates"] == 15000 and config["disable_early_stopping"]
    assert records[-1]["optimizer_updates"] == 15000
    assert config["max_seq_length"] == 64
    assert config["hparams"]["warmup_steps"] == 400
    batch_size = config["hparams"].get("bs", config["hparams"].get("batch_size"))
    assert batch_size * config["hparams"]["grad_accum"] == 128
    assert torch.cuda.is_available(), "GPU audit required"
    device = torch.device("cuda")
    items = stratified_subsample(get_train("sst2"), 24000, "label", seed=42)
    evals = get_evals("sst2")
    assert list(evals) == ["validation"] and len(evals["validation"]) == 872
    if args.output.exists():
        assert args.reuse_verified, "Refuse to overwrite an existing audit"
        previous = json.loads(args.output.read_text())
        assert previous["source_run"] == str(args.run_dir)
        assert previous["config"] == config and previous["best_info"] == best
        assert previous["train_sha256"] == digest(items)
        assert previous["validation_sha256"] == digest(evals["validation"])
        assert previous["source_hashes"] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in HERE.glob("*.py")}
        assert previous["dev_reproduced"] and previous["strong_train_fit"]
        print("[REUSE VERIFIED] " + str(args.output), flush=True)
        return
    msl = config["max_seq_length"]
    ckpt = str(args.run_dir / "ckpt")
    if config["model"] == "pixel-base":
        assert config["renderer"] == "PangoCairoTextRenderer"
        processor = PangoCairoTextRenderer.from_pretrained(config["model_path"], rgb=True)
        processor.max_seq_length = msl
        transforms = get_transforms(do_resize=True, size=(processor.pixels_per_patch,
                                                       processor.pixels_per_patch * msl))
        model, loading = PIXELForSequenceClassification.from_pretrained(
            ckpt, pooling_mode=PoolingMode.MEAN, add_layer_norm=True,
            output_loading_info=True)
        correct_size = [processor.pixels_per_patch, processor.pixels_per_patch * msl]
        model.config.image_size = correct_size
        model.image_size = correct_size
        model.vit.embeddings.patch_embeddings.image_size = correct_size
        dataset = lambda data: PixelDataset(data, processor, transforms, msl, ("text", None))
    else:
        assert config["model"] == "bert"
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            ckpt, output_loading_info=True)
        dataset = lambda data: TextDataset(data, tokenizer, msl, ("text", None))
    assert not loading.get("missing_keys"), loading
    assert not loading.get("unexpected_keys"), loading
    assert not loading.get("mismatched_keys"), loading
    model.to(device)
    report = {"source_run": str(args.run_dir), "config": config, "best_info": best,
              "train_sha256": digest(items), "validation_sha256": digest(evals["validation"]),
              "train_label_counts": dict(Counter(i["label"] for i in items)),
              "validation_label_counts": dict(Counter(i["label"] for i in evals["validation"])),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "transformers": transformers.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name()},
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (HERE / "source_snapshot").glob("*.py")},
              "online_final": records[-1]}
    report["train"] = measure(model, dataset(items), device)
    # Save only aggregate train measurements, keeping the validation predictions for inspection.
    for key in ("labels", "predictions", "probabilities"):
        report["train"].pop(key)
    report["validation"] = measure(model, dataset(evals["validation"]), device)
    report["dev_reproduced"] = abs(report["validation"]["accuracy"] - best["value"]) < 1e-7
    report["strong_train_fit"] = report["train"]["accuracy"] >= 0.98
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": str(args.run_dir), "train": report["train"],
                      "dev": report["validation"]["accuracy"],
                      "reproduced": report["dev_reproduced"]}), flush=True)
    assert report["dev_reproduced"], "Saved checkpoint does not reproduce recorded best-dev"
    if not args.allow_low_fit:
        assert report["strong_train_fit"], "Online fit was not confirmed by checkpoint audit"


if __name__ == "__main__":
    main()
