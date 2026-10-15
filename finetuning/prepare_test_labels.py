"""Map official Stanford binary root labels to existing GLUE SST-2 text.

Do not replace test text, use predictions, or impute labels. Require a unique
mapping for every row, exact label agreement on the known validation split,
and equal counts with the author's binary test split.
"""
import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_utils import get_train, get_evals, SST2_DIR

# Reviewed individually against the author's tree roots on 2026-10-03.
# Only these four known text variants are admitted, with both full sentences
# pinned. No fuzzy match or prediction is used to assign an evaluation label.
MANUAL_TEST_MAPPING = {
    71: {
        "glue_text": "a very witty take on change , risk and romance , and the film uses humor to make its points about acceptance and growth .",
        "author_text": "A very witty take on change , risk and romance , and the film uses humour to make its points about acceptance and growth .",
        "label": 1, "difference": "humour/humor",
    },
    112: {
        "glue_text": "he has not learned that storytelling is what the movies are about .",
        "author_text": "He has not learnt that storytelling is what the movies are about .",
        "label": 0, "difference": "learnt/learned",
    },
    186: {
        "glue_text": "a recent favorite at sundance , this white-trash satire will inspire the affection of even those unlucky people who never owned a cassette of def leppard 's pyromania .",
        "author_text": "A recent favourite at Sundance , this white-trash satire will inspire the affection of even those unlucky people who never owned a cassette of Def Leppard 's Pyromania .",
        "label": 1, "difference": "favourite/favorite",
    },
    730: {
        "glue_text": "a frantic search for laughs , with a hit-to-miss ratio that does n't exactly favor the audience .",
        "author_text": "A frantic search for laughs , with a hit-to-miss ratio that does n't exactly favour the audience .",
        "label": 0, "difference": "favour/favor",
    },
}


def canonical(text):
    text = html.unescape(text).casefold()
    text = text.replace("\\/", "/").replace("\\*", "*")
    for source, target in (("-lrb-", "("), ("-rrb-", ")"),
                           ("-lsb-", "["), ("-rsb-", "]"),
                           ("-lcb-", "{"), ("-rcb-", "}")):
        text = text.replace(source, target)
    return re.sub(r"\s+", "", text)


def read_roots(archive, split):
    items = []
    with ZipFile(archive) as zipped:
        for tree in zipped.read(f"trees/{split}.txt").decode("utf-8").splitlines():
            root = int(re.match(r"\(([0-4])\s", tree).group(1))
            if root == 2:
                continue
            leaves = re.findall(r"\([0-4]\s+([^()]+)\)", tree)
            assert leaves
            items.append({"text": " ".join(leaves), "label": int(root >= 3)})
    return items


def mapped_labels(items, official, manual=None):
    lookup = {}
    for item in official:
        lookup.setdefault(canonical(item["text"]), set()).add(item["label"])
    labels = []
    used = set()
    for index, item in enumerate(items):
        choices = lookup.get(canonical(item["text"]), set())
        if not choices and manual is not None and index in manual:
            reviewed = manual[index]
            assert item["text"] == reviewed["glue_text"], "Manual mapping target text changed"
            choices = lookup.get(canonical(reviewed["author_text"]), set())
            assert choices == {reviewed["label"]}, "Manual mapping disagrees with author root label"
            used.add(index)
        assert len(choices) == 1, f"Missing/ambiguous root mapping: {item['text']!r} -> {choices}"
        labels.append(next(iter(choices)))
    assert len(items) == len(official)
    if manual is not None:
        assert used == set(manual), "Manual exception list no longer matches source data"
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=HERE / "test_data/trainDevTestTrees_PTB.zip")
    parser.add_argument("--output", type=Path, default=HERE / "test_data/sst2_test_labeled.json")
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to replace prepared test labels"
    val = get_evals("sst2")["validation"]
    labels = mapped_labels(val, read_roots(args.archive, "dev"))
    assert labels == [i["label"] for i in val], "Author label mapping disagrees with known validation labels"
    df = pd.read_parquet(Path(SST2_DIR) / "test-00000-of-00001.parquet")
    assert set(df["label"]) == {-1}, "Unexpected existing test labels; inspect manually"
    test = [{"text": str(text).strip()} for text in df["sentence"]]
    labels = mapped_labels(test, read_roots(args.archive, "test"), MANUAL_TEST_MAPPING)
    assert len(test) == 1821
    train_keys = {canonical(i["text"]) for i in get_train("sst2")}
    val_keys = {canonical(i["text"]) for i in val}
    for row, label in zip(test, labels):
        row["label"] = label
    train_overlap = [i for i, row in enumerate(test) if canonical(row["text"]) in train_keys]
    val_overlap = [i for i, row in enumerate(test) if canonical(row["text"]) in val_keys]
    report = {"source_url": "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
              "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              "test_parquet_sha256": hashlib.sha256((Path(SST2_DIR) / "test-00000-of-00001.parquet").read_bytes()).hexdigest(),
              "validation_label_agreement": "872/872", "test_mapping": "1821/1821",
              "test_mapping_details": "1817 normalized exact matches; 4 individually reviewed British/American variants",
              "reviewed_test_variants": MANUAL_TEST_MAPPING,
              "label_counts": dict(Counter(labels)),
              "mapping_normalization": "HTML unescape, casefold, PTB bracket aliases and escaped slash/asterisk, remove whitespace; original inference text is unchanged",
              "overlap_with_fixed_24k_train_indices": train_overlap,
              "overlap_with_validation_indices": val_overlap,
              "items": test}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "items"}), flush=True)


if __name__ == "__main__":
    main()
