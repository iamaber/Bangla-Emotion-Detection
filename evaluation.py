"""Shared evaluation protocol for the Bangla emotion notebooks.

Every notebook uses the same 70/15/15 split, the same metric definitions and
the same majority baselines, so results are comparable across models.

Protocol
--------
train (70%)  fit vectorizers/tokenizers and model weights
val   (15%)  epoch/threshold selection only (early stopping, F1 sweep)
test  (15%)  evaluated exactly once per model, never used for tuning

Metric definitions (always reported by name, never as a bare "accuracy")
------------------------------------------------------------------------
emotions (multi-label, 6 independent binary labels)
    per-label precision/recall/f1, macro_f1, micro_f1,
    subset_accuracy (all 6 labels must match), hamming_loss
topic (single-label multiclass, 12 classes)
    accuracy (exact topic match), macro_f1, per-class precision/recall/f1
"""

import json
import os
import random

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

SEED = 42

EMOTIONS = ["Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]

TEST_SIZE = 0.15
VAL_SIZE = 0.15


def set_global_seed(seed=SEED):
    """Seed python/numpy and (if installed) TensorFlow.

    ``PYTHONHASHSEED`` only takes effect if it is set *before* the interpreter
    starts, so run the notebooks with ``PYTHONHASHSEED=<seed>`` (see README);
    setting it here only propagates the value to child processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.keras.utils.set_random_seed(seed)


def _strata(y_labels, min_count=5, rare="RARE"):
    """Turn multi-label rows into single-label strata for sklearn splits.

    Rows sharing a label signature are stratified together. Signatures rarer
    than ``min_count`` are merged into one ``RARE`` stratum so that
    ``train_test_split`` never receives a class with a single member.
    """
    signatures = np.array(["".join(str(int(v)) for v in row) for row in y_labels])
    values, counts = np.unique(signatures, return_counts=True)
    rare_signatures = set(values[counts < min_count])
    return np.array([rare if s in rare_signatures else s for s in signatures])


def split_indices(y_labels, seed=SEED, test_size=TEST_SIZE, val_size=VAL_SIZE):
    """Return train/val/test positional indices from a multi-label-stratified split.

    ``y_labels`` is the (n_samples, n_labels) 0/1 emotion matrix. One split is
    computed and shared by every target, so the same rows land in the same
    split for emotions and topic.
    """
    y_labels = np.asarray(y_labels)
    all_idx = np.arange(y_labels.shape[0])
    strata = _strata(y_labels)

    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=seed, stratify=strata
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=strata[train_val_idx],
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def best_thresholds(y_true, scores, candidates=None):
    """Pick a per-label decision threshold by maximising F1 on the validation set.

    Called on validation data only; the chosen thresholds are then frozen and
    applied to the test set, which is not used for this decision.
    """
    from sklearn.metrics import f1_score

    if candidates is None:
        candidates = np.arange(0.05, 1.0, 0.05)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    thresholds = []
    for i in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for threshold in candidates:
            predicted = (scores[:, i] >= threshold).astype(int)
            f1 = f1_score(y_true[:, i], predicted, zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(threshold), f1
        thresholds.append(best_t)
    return np.array(thresholds)


def emotions_metrics(y_true, y_pred, labels=EMOTIONS):
    """Multi-label metrics for the emotion targets (predictions already binarized)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        "task": "emotions",
        "definition": "multi-label binary; subset_accuracy = all 6 labels exact",
        "per_label": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def topic_metrics(y_true_codes, y_pred_codes, class_names):
    """Single-label multiclass metrics for the topic target.

    ``y_true_codes``/``y_pred_codes`` are integer label codes and
    ``class_names`` maps each code to its topic name.
    """
    y_true_codes = np.asarray(y_true_codes)
    y_pred_codes = np.asarray(y_pred_codes)
    codes = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_codes, y_pred_codes, labels=codes, average=None, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_true_codes, y_pred_codes, labels=codes, average="macro", zero_division=0
    )[2]
    return {
        "task": "topic",
        "definition": "single-label multiclass; accuracy = exact topic match",
        "accuracy": float(accuracy_score(y_true_codes, y_pred_codes)),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


def majority_baseline(y_true, labels=EMOTIONS):
    """Trivial per-label majority classifier: the floor every model must beat.

    No emotion exceeds 50% prevalence, so the per-label majority is all-zero.
    Also reported: the most frequent full label signature, which is the
    strongest constant predictor for subset accuracy.
    """
    y_true = np.asarray(y_true)
    marginals = y_true.mean(axis=0)
    majority = (marginals > 0.5).astype(int)
    y_pred = np.tile(majority, (y_true.shape[0], 1))
    _, counts = np.unique(y_true, axis=0, return_counts=True)
    return {
        "task": "emotions",
        "definition": "per-label majority classifier (no model, no features)",
        "per_label_accuracy": {
            label: float(np.mean(y_true[:, i] == majority[i]))
            for i, label in enumerate(labels)
        },
        "per_label_majority_accuracy": float(np.mean(y_true == y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "most_frequent_signature_subset_accuracy": float(counts.max() / len(y_true)),
        "label_marginals": {
            label: float(marginals[i]) for i, label in enumerate(labels)
        },
    }


def save_metrics(path, payload):
    """Persist metrics as JSON so numbers are artifacts, not notebook output."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path
