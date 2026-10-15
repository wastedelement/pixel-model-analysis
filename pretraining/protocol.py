"""Fixed settings and outcome definition for the MNLI intervention."""
import hashlib
import json
import math

SEEDS = (100, 5678)
INITIALIZATIONS = ("original", "pawsx", "taxi1500")
TRAIN_N = 24000
BATCH_SIZE = 32
GRAD_ACCUM = 4
SCREEN_EPOCHS = 10
SCHEDULE_EPOCHS = 80
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 3e-5
EXPECTED_TARGET = "912998ab93f6073e7d087f607ed01d641a4e47259767124bc8c0cdca045b951d"
EXPECTED_DEVS = {
    "dev_matched": "210c22b8dfaef898f462e9918f7d87e72d46a236f035e6926bac0ee625246f22",
    "dev_mismatched": "afdd5565414cea51ff3b8278427fb71714bd6c5bffb971f75934580de2def3a0",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def warmup_steps(seed):
    if seed not in SEEDS:
        raise ValueError("Only the two selected case-study seeds are supported")
    return 100 if seed == 100 else 0


def updates_per_epoch():
    return math.ceil(math.ceil(TRAIN_N / BATCH_SIZE) / GRAD_ACCUM)


def training_fit(record):
    """Original operational criterion, evaluated on online training metrics."""
    return record["train_loss"] < 1.0 and record["train_acc"] > 0.5
