"""Multiclass Matthews Correlation Coefficient (sklearn-free)."""
import numpy as np


def matthews_corrcoef(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    C = np.zeros((K, K), dtype=np.float64)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            C[i, j] = float(np.sum((y_true == c1) & (y_pred == c2)))
    t = C.sum(axis=1)  # row sums = true class counts
    p = C.sum(axis=0)  # col sums = pred class counts
    c = float(np.trace(C))
    s = float(C.sum())
    num = c * s - float(np.dot(t, p))
    den = np.sqrt((s * s - float(np.dot(p, p))) * (s * s - float(np.dot(t, t))))
    if den == 0:
        return 0.0
    return float(num / den)


def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
        rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0
