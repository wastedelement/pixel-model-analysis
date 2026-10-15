import csv
import os
import random
from collections import defaultdict, OrderedDict
from pathlib import Path

HF_CACHE_ROOT = Path(os.environ.get("ROBUSTNESS_HF_CACHE", ".cache/huggingface"))
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))

from datasets import load_dataset, load_from_disk

DATA_ROOT = Path(os.environ.get("XLING_DATA_DIR", "data/xling"))
TAXI1500_LABELED_ROOT = Path(os.environ.get("TAXI1500_LABELED_DIR", str(DATA_ROOT / "taxi-1500-labeled")))
SEED = 42

LANGS_8 = OrderedDict([
    ("en", {"name": "English", "xnli": "en", "sib": "eng_Latn", "taxi": "eng_data", "paws": "en", "group": "source"}),
    ("de", {"name": "German", "xnli": "de", "sib": "deu_Latn", "taxi": "deu_deu1912", "paws": "de", "group": "close"}),
    ("fr", {"name": "French", "xnli": "fr", "sib": "fra_Latn", "taxi": "fra_francl", "paws": "fr", "group": "close"}),
    ("es", {"name": "Spanish", "xnli": "es", "sib": "spa_Latn", "taxi": "spa_spabes", "paws": "es", "group": "close"}),
    ("ru", {"name": "Russian", "xnli": "ru", "sib": "rus_Cyrl", "taxi": "rus_russyn", "paws": None, "group": "cyrillic"}),
    ("zh", {"name": "Chinese", "xnli": "zh", "sib": "zho_Hans", "taxi": "cmn_cmn-cu89s", "paws": "zh", "group": "far"}),
    ("ja", {"name": "Japanese", "xnli": None, "sib": "jpn_Jpan", "taxi": None, "paws": "ja", "group": "far"}),
    ("ko", {"name": "Korean", "xnli": None, "sib": "kor_Hang", "taxi": None, "paws": "ko", "group": "far"}),
])

TASK_LANGS = {
    "xnli": ("en", "de", "fr", "es", "ru", "zh"),
    "sib200": ("en", "de", "fr", "es", "ru", "zh", "ko"),
    "taxi1500": ("en", "de", "fr", "ru", "zh"),
    "pawsx": ("en", "de", "fr", "es", "zh", "ja", "ko"),
}

XNLI_LABELS = ["entailment", "neutral", "contradiction"]
SIB_LABELS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
TAXI_LABELS = ["faith", "grace", "sin", "violence", "description", "recommendation"]
PAWSX_LABELS = ["not_paraphrase", "paraphrase"]
SIB_LABEL2ID = {v: i for i, v in enumerate(SIB_LABELS)}
TAXI_LABEL2ID = {v: i for i, v in enumerate(TAXI_LABELS)}

TASK_DATA_PLAN = {
    "xnli": {
        "num_labels": 3,
        "label_names": XNLI_LABELS,
        "text_keys": ("premise", "hypothesis"),
        "metric": "acc",
        "primary_eval": "en_validation",
        "train_n": 20000,
        "eval_n": 2000,
    },
    "sib200": {
        "num_labels": 7,
        "label_names": SIB_LABELS,
        "text_keys": ("text", None),
        "metric": "f1",
        "primary_eval": "en_dev",
        "train_n": None,
        "eval_n": None,
    },
    "taxi1500": {
        "num_labels": 6,
        "label_names": TAXI_LABELS,
        "text_keys": ("text", None),
        "metric": "f1",
        "paper_metric": "macro_f1",
        "primary_eval": "en_dev",
        "train_n": 860,
        "dev_n": 106,
        "test_n": 111,
        "note": "requires prepared labeled Taxi1500 TSVs under TAXI1500_LABELED_DIR",
    },
    "pawsx": {
        "num_labels": 2,
        "label_names": PAWSX_LABELS,
        "text_keys": ("sentence1", "sentence2"),
        "metric": "acc",
        "primary_eval": "en_validation",
        "train_n": None,
        "eval_n": None,
    },
}


def stratified_subsample(items, n, label_key="label", seed=SEED):
    if n is None or n >= len(items):
        return list(items)
    by_label = defaultdict(list)
    for idx, it in enumerate(items):
        by_label[it[label_key]].append(idx)
    rng = random.Random(seed)
    for idxs in by_label.values():
        rng.shuffle(idxs)
    total = len(items)
    chosen = []
    for idxs in by_label.values():
        k = int(round(n * len(idxs) / total))
        chosen.extend(idxs[:k])
    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
    elif len(chosen) < n:
        used = set(chosen)
        rest = [i for i in range(len(items)) if i not in used]
        rng.shuffle(rest)
        chosen.extend(rest[:n - len(chosen)])
    rng.shuffle(chosen)
    return [items[i] for i in chosen]


def simple_subsample(items, n, seed=SEED):
    if n is None or n >= len(items):
        return list(items)
    idxs = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    return [items[i] for i in idxs[:n]]


def _xnli_file(lang_key, split):
    lang = LANGS_8[lang_key]["xnli"]
    if lang is None:
        raise KeyError(f"XNLI does not provide language {lang_key!r}")
    path = DATA_ROOT / "xnli" / lang
    return next(path.glob(f"{split}-*.parquet"))


def load_xnli(lang_key, split):
    ds = load_dataset("parquet", data_files=str(_xnli_file(lang_key, split)), split="train")
    out = []
    for r in ds:
        out.append({
            "premise": str(r["premise"]).strip(),
            "hypothesis": str(r["hypothesis"]).strip(),
            "label": int(r["label"]),
            "lang": lang_key,
        })
    return out


def load_sib200(lang_key, split):
    lang = LANGS_8[lang_key]["sib"]
    split_name = "dev" if split == "validation" else split
    path = DATA_ROOT / "sib-200" / "data" / lang / f"{split_name}.tsv"
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.append({
                "text": str(r["text"]).strip(),
                "label": SIB_LABEL2ID[str(r["category"]).strip()],
                "lang": lang_key,
            })
    return out


def has_taxi1500_labeled():
    return all(
        (TAXI1500_LABELED_ROOT / lang_key / f"{split}.tsv").exists()
        for lang_key in TASK_LANGS["taxi1500"]
        for split in ("train", "dev", "test")
    )


def require_taxi1500_labeled():
    if not has_taxi1500_labeled():
        raise FileNotFoundError(
            "Taxi1500 zero-shot topic classification requires the labeled Taxi1500 dataset "
            "(6 topics, 860/106/111 split per language). Provide the original processed TSV files "
            "or set TAXI1500_LABELED_DIR to a compatible processed TSV root."
        )


def load_taxi1500(lang_key, split):
    require_taxi1500_labeled()
    path = TAXI1500_LABELED_ROOT / lang_key / f"{split}.tsv"
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.append({
                "text": str(r["text"]).strip(),
                "label": TAXI_LABEL2ID[str(r["label"]).strip().lower()],
                "lang": lang_key,
                "id": str(r["id"]).strip(),
            })
    return out


def load_pawsx(lang_key, split):
    lang = LANGS_8[lang_key]["paws"]
    if lang is None:
        raise KeyError(f"PAWS-X does not provide language {lang_key!r}")
    path = DATA_ROOT / "paws-x" / lang / f"{split}.parquet"
    ds = load_dataset("parquet", data_files=str(path), split="train")
    return [
        {
            "sentence1": str(row["sentence1"]).strip(),
            "sentence2": str(row["sentence2"]).strip(),
            "label": int(row["label"]),
            "lang": lang_key,
        }
        for row in ds
    ]


def get_train(task):
    if task == "xnli":
        return stratified_subsample(load_xnli("en", "train"), TASK_DATA_PLAN[task]["train_n"], seed=SEED)
    if task == "sib200":
        return load_sib200("en", "train")
    if task == "taxi1500":
        return load_taxi1500("en", "train")
    if task == "pawsx":
        return load_pawsx("en", "train")
    raise ValueError(task)


def get_train_evals(task):
    if task == "xnli":
        items = stratified_subsample(load_xnli("en", "validation"), TASK_DATA_PLAN[task]["eval_n"], seed=SEED + 1)
        return {"en_validation": items}
    if task == "sib200":
        return {"en_dev": load_sib200("en", "dev")}
    if task == "taxi1500":
        return {"en_dev": load_taxi1500("en", "dev")}
    if task == "pawsx":
        return {"en_validation": load_pawsx("en", "validation")}
    raise ValueError(task)


def get_eval_langs(task):
    return OrderedDict((k, LANGS_8[k]) for k in TASK_LANGS[task])


def get_final_evals(task):
    if task == "xnli":
        out = OrderedDict()
        for lang_key in get_eval_langs(task).keys():
            items = stratified_subsample(load_xnli(lang_key, "test"), TASK_DATA_PLAN[task]["eval_n"], seed=SEED + 10)
            out[f"{lang_key}_test"] = items
        return out
    if task == "sib200":
        out = OrderedDict()
        for lang_key in get_eval_langs(task).keys():
            out[f"{lang_key}_test"] = load_sib200(lang_key, "test")
        return out
    if task == "taxi1500":
        out = OrderedDict()
        for lang_key in get_eval_langs(task).keys():
            out[f"{lang_key}_test"] = load_taxi1500(lang_key, "test")
        return out
    if task == "pawsx":
        out = OrderedDict()
        for lang_key in get_eval_langs(task).keys():
            out[f"{lang_key}_test"] = load_pawsx(lang_key, "test")
        return out
    raise ValueError(task)


def task_summary(task):
    if task == "taxi1500" and not has_taxi1500_labeled():
        plan = TASK_DATA_PLAN[task]
        return {
            "task": task,
            "available": False,
            "note": plan["note"],
            "planned_train": plan["train_n"],
            "planned_dev": plan["dev_n"],
            "planned_test": plan["test_n"],
            "num_labels": plan["num_labels"],
            "label_names": plan["label_names"],
            "languages": get_eval_langs(task),
        }
    train = get_train(task)
    train_evals = get_train_evals(task)
    final_evals = get_final_evals(task)
    return {
        "task": task,
        "available": True,
        "train": len(train),
        "train_evals": {k: len(v) for k, v in train_evals.items()},
        "final_evals": {k: len(v) for k, v in final_evals.items()},
        "num_labels": TASK_DATA_PLAN[task]["num_labels"],
        "label_names": TASK_DATA_PLAN[task]["label_names"],
        "languages": get_eval_langs(task),
    }


if __name__ == "__main__":
    import json
    for t in TASK_DATA_PLAN:
        print(json.dumps(task_summary(t), ensure_ascii=False, indent=2))
