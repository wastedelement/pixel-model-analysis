"""Common dataset loading + stratified subsampling for SST-2 / CoLA / MNLI."""
import os
import json
import random
from collections import defaultdict


SST2_DIR = os.environ.get("SST2_DATA_DIR", "data/sst2")
COLA_DIR = os.environ.get("COLA_DATA_DIR", "data/cola")
MNLI_DIR = os.environ.get("MNLI_DATA_DIR", "data/mnli")

SEED = 42

# label index conventions
SST2_NUM_LABELS = 2  # 0 = neg, 1 = pos
COLA_NUM_LABELS = 2  # 0 = unacceptable, 1 = acceptable
MNLI_LABELS = ["entailment", "neutral", "contradiction"]
MNLI_LABEL2ID = {l: i for i, l in enumerate(MNLI_LABELS)}
MNLI_NUM_LABELS = 3

TASK_INFO = {
    "sst2": {"num_labels": 2, "is_pair": False, "metric": "accuracy"},
    "cola": {"num_labels": 2, "is_pair": False, "metric": "mcc"},
    "mnli": {"num_labels": 3, "is_pair": True,  "metric": "accuracy"},
}


def stratified_subsample(items, n, label_key="label", seed=SEED):
    """Sample n items keeping label distribution."""
    if n is None or n >= len(items):
        return list(items)
    by_label = defaultdict(list)
    for it in items:
        by_label[it[label_key]].append(it)
    rng = random.Random(seed)
    for k in by_label:
        rng.shuffle(by_label[k])
    total = len(items)
    out = []
    for label, lst in by_label.items():
        k = int(round(n * len(lst) / total))
        out.extend(lst[:k])
    # tweak if rounding lost/gained
    if len(out) > n:
        out = out[:n]
    elif len(out) < n:
        # pad from leftover pool
        rng.shuffle(out)
        rest = [it for it in items if it not in out]
        out += rest[:(n - len(out))]
    rng.shuffle(out)
    return out


# ---------- SST-2 ----------
def load_sst2(split):
    """split in {train, validation}. Returns list of {text, label}."""
    fname = {
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet",
    }[split]
    import pandas as pd
    df = pd.read_parquet(os.path.join(SST2_DIR, fname))
    return [
        {"text": str(r["sentence"]).strip(), "label": int(r["label"])}
        for _, r in df.iterrows()
    ]


# ---------- CoLA ----------
def load_cola(split):
    """split in {train, dev_in, dev_ood}. Returns list of {text, label}."""
    fname = {
        "train":   "in_domain_train.tsv",
        "dev_in":  "in_domain_dev.tsv",
        "dev_ood": "out_of_domain_dev.tsv",
    }[split]
    out = []
    with open(os.path.join(COLA_DIR, fname)) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            # cols: source, label, ?, sentence
            if len(parts) < 4:
                continue
            label = int(parts[1])
            text = parts[3].strip()
            out.append({"text": text, "label": label})
    return out


# ---------- MNLI ----------
def load_mnli(split):
    """split in {train, dev_matched, dev_mismatched}.
    Returns list of {premise, hypothesis, label}."""
    fname = {
        "train":           "multinli_1.0_train.jsonl",
        "dev_matched":     "multinli_1.0_dev_matched.jsonl",
        "dev_mismatched":  "multinli_1.0_dev_mismatched.jsonl",
    }[split]
    out = []
    with open(os.path.join(MNLI_DIR, fname)) as f:
        for line in f:
            d = json.loads(line)
            gold = d.get("gold_label", "-")
            if gold not in MNLI_LABEL2ID:
                continue   # skip "-" no consensus
            out.append({
                "premise": d["sentence1"].strip(),
                "hypothesis": d["sentence2"].strip(),
                "label": MNLI_LABEL2ID[gold],
            })
    return out


# ---------- task -> (train, eval-splits) ----------
TASK_DATA_PLAN = {
    "sst2": {
        "train_n": 24000,
        "train_loader": ("sst2", "train"),
        "eval_splits": [("validation", ("sst2", "validation"), None)],  # (name, loader, subsample_n)
        "is_pair": False,
        "text_keys": ("text", None),
        "num_labels": 2,
    },
    "cola": {
        "train_n": None,  # use all 8551
        "train_loader": ("cola", "train"),
        "eval_splits": [
            ("in_domain_dev",  ("cola", "dev_in"),  None),
            ("ood_dev",        ("cola", "dev_ood"), None),
        ],
        "is_pair": False,
        "text_keys": ("text", None),
        "num_labels": 2,
    },
    "mnli": {
        "train_n": 24000,
        "train_loader": ("mnli", "train"),
        "eval_splits": [
            ("dev_matched",     ("mnli", "dev_matched"),    2000),
            ("dev_mismatched",  ("mnli", "dev_mismatched"), 2000),
        ],
        "is_pair": True,
        "text_keys": ("premise", "hypothesis"),
        "num_labels": 3,
    },
}


_LOADERS = {"sst2": load_sst2, "cola": load_cola, "mnli": load_mnli}


def get_train(task):
    plan = TASK_DATA_PLAN[task]
    name, split = plan["train_loader"]
    items = _LOADERS[name](split)
    items = stratified_subsample(items, plan["train_n"], "label", SEED)
    return items


def get_evals(task):
    """Returns dict {split_name: list_of_items}."""
    plan = TASK_DATA_PLAN[task]
    out = {}
    for split_name, (loader_name, sub), n in plan["eval_splits"]:
        items = _LOADERS[loader_name](sub)
        if n is not None:
            items = stratified_subsample(items, n, "label", SEED + 1)
        out[split_name] = items
    return out


if __name__ == "__main__":
    # quick sanity test
    for task in ("sst2", "cola", "mnli"):
        tr = get_train(task)
        evs = get_evals(task)
        print(f"[{task}] train={len(tr)}  evals=" + ", ".join(f"{k}={len(v)}" for k, v in evs.items()))
        from collections import Counter
        print(f"    train label dist: {Counter(it['label'] for it in tr)}")
