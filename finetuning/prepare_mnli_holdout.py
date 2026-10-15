"""Rebuild the recorded MNLI selection/holdout split without training."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from data_utils import get_train, get_evals, load_mnli, stratified_subsample


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pair(item):
    return item['premise'], item['hypothesis']


def prepare():
    selection = get_evals('mnli')
    train = get_train('mnli')
    if digest(train) != '912998ab93f6073e7d087f607ed01d641a4e47259767124bc8c0cdca045b951d':
        raise ValueError('Historical training subset differs: check original data and row order')
    excluded = {pair(x) for x in train}
    for items in selection.values():
        excluded.update(pair(x) for x in items)
    heldout, audit = {}, {}
    for offset, split in enumerate(('dev_matched', 'dev_mismatched')):
        full = load_mnli(split)
        counts = collections.Counter(pair(x) for x in full)
        remaining = [x for x in full if pair(x) not in excluded and counts[pair(x)] == 1]
        if len(remaining) < 2000:
            raise ValueError('Insufficient unique holdout examples')
        chosen = stratified_subsample(remaining, 2000, 'label', 20260928 + offset)
        if {pair(x) for x in chosen} & excluded or len({pair(x) for x in chosen}) != 2000:
            raise ValueError('Holdout overlap detected')
        heldout[split] = chosen
        excluded.update(pair(x) for x in chosen)
        audit[split] = {'full_n': len(full), 'selection_n': len(selection[split]),
                        'eligible_n': len(remaining), 'holdout_n': len(chosen),
                        'holdout_sha256': digest(chosen),
                        'selection_sha256': digest(selection[split]),
                        'label_counts': dict(collections.Counter(x['label'] for x in chosen))}
    return {'purpose': 'held-out development complement, not official test',
            'selection_rule': 'best matched-selection-development checkpoint; no holdout selection',
            'train_subset_sha256': digest(train), 'audit': audit, 'holdout': heldout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/mnli_holdout/manifest.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing manifest')
    result = prepare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result['audit'], indent=2))


if __name__ == '__main__':
    main()
