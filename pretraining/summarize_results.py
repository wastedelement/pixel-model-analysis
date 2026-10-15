"""Inspect the numeric supplement, or aggregate fresh MNLI completion files."""
import argparse
import json
from pathlib import Path
from protocol import SEEDS, INITIALIZATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path)
    args = parser.parse_args()
    if args.runs_root is None:
        payload = json.loads(Path(__file__).with_name("recorded_results.json").read_text())
        print("SST-2 development accuracy (%)")
        for record in payload["sst2"]:
            print(f"  {record['model']:10s} {record['state']:8s} {record['development_accuracy']:.2f}")
        runs = payload["mnli"]
    else:
        runs = []
        for path in sorted(args.runs_root.glob("mnli_pixel-base_*/completion.json")):
            record = json.loads(path.read_text())
            runs.append({"seed": record["seed"], "repeat": record["repeat"],
                         "initialization": record["initialization"], "success": record["fit_epoch10"]})
    print("MNLI successful runs / completed repetitions")
    seen = set()
    for record in runs:
        key = (record["seed"], record["initialization"], record["repeat"])
        if key in seen:
            raise ValueError("Duplicate run identifier")
        if key[0] not in SEEDS or key[1] not in INITIALIZATIONS or key[2] not in range(1, 6):
            raise ValueError("Run identifier does not belong to the fixed protocol")
        seen.add(key)
    for seed in SEEDS:
        for initialization in INITIALIZATIONS:
            group = [r for r in runs if r["seed"] == seed and r["initialization"] == initialization]
            print(f"  seed={seed:<4d} {initialization:10s} {sum(r['success'] for r in group)}/{len(group)}")
    for initialization in INITIALIZATIONS:
        group = [r for r in runs if r["initialization"] == initialization]
        print(f"  combined  {initialization:10s} {sum(r['success'] for r in group)}/{len(group)}")


if __name__ == "__main__":
    main()
