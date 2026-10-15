"""Evaluate validation-selected checkpoints on mapped author test labels.

Test accuracy is never used for checkpoint selection or training changes.
"""
import argparse
import json
from pathlib import Path

from audit_sst2_checkpoint import (
    torch, digest, measure, PixelDataset, PangoCairoTextRenderer,
    PIXELForSequenceClassification, PoolingMode, get_transforms,
    AutoTokenizer, AutoModelForSequenceClassification, TextDataset,
    get_evals,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, default=Path(__file__).resolve().parent / "test_data/sst2_test_labeled.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to overwrite test measurements"
    config = json.loads((args.run_dir / "run_config.json").read_text())
    best = json.loads((args.run_dir / "ckpt/best_info.json").read_text())
    assert config["task"] == "sst2" and config["max_seq_length"] == 64
    assert config["train_n"] in (2000, 8000, 24000) and config["max_updates"] == 15000
    assert torch.cuda.is_available()
    test = json.loads(args.test_data.read_text())
    assert test["validation_label_agreement"] == "872/872" and test["test_mapping"] == "1821/1821"
    ckpt = str(args.run_dir / "ckpt")
    msl = config["max_seq_length"]
    if config["model"] == "pixel-base":
        assert config["renderer"] == "PangoCairoTextRenderer"
        processor = PangoCairoTextRenderer.from_pretrained(config["model_path"], rgb=True)
        processor.max_seq_length = msl
        transforms = get_transforms(do_resize=True, size=(processor.pixels_per_patch,
                                                       processor.pixels_per_patch * msl))
        model, loading = PIXELForSequenceClassification.from_pretrained(
            ckpt, pooling_mode=PoolingMode.MEAN, add_layer_norm=True, output_loading_info=True)
        size = [processor.pixels_per_patch, processor.pixels_per_patch * msl]
        model.config.image_size = size
        model.image_size = size
        model.vit.embeddings.patch_embeddings.image_size = size
        dataset = lambda rows: PixelDataset(rows, processor, transforms, msl, ("text", None))
    else:
        assert config["model"] == "bert"
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model, loading = AutoModelForSequenceClassification.from_pretrained(ckpt, output_loading_info=True)
        dataset = lambda rows: TextDataset(rows, tokenizer, msl, ("text", None))
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not loading.get(name), loading
    model.to("cuda")
    validation = measure(model, dataset(get_evals("sst2")["validation"]), "cuda")
    assert abs(validation["accuracy"] - best["value"]) < 1e-7, "Validation sanity check failed"
    results = measure(model, dataset(test["items"]), "cuda")
    excluded = set(test["overlap_with_fixed_24k_train_indices"] + test["overlap_with_validation_indices"])
    keep = [i for i in range(len(test["items"])) if i not in excluded]
    disjoint_acc = sum(results["labels"][i] == results["predictions"][i] for i in keep) / len(keep)
    report = {"source_run": str(args.run_dir), "config": config, "best_info": best,
              "selection": "Best recorded validation accuracy only; no test-based selection",
              "test_data_sha256": digest(test["items"]),
              "test_provenance": {k: v for k, v in test.items() if k != "items"},
              "validation_reproduced": validation["accuracy"], "test": results,
              "disjoint_test": {"n": len(keep), "accuracy": disjoint_acc,
                                "excluded_indices": sorted(excluded)},
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": str(args.run_dir), "dev": validation["accuracy"],
                      "test": results["accuracy"], "disjoint_test": disjoint_acc}), flush=True)


if __name__ == "__main__":
    main()
