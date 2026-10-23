"""Tests for the evaluation protocol and for padding masking.

Run with:  python -m unittest discover -s tests -v

The masking test is the regression guard for the padded-sequence bug: without
masking, the same comment with more `0` padding produces a different pooled
representation, so padding leaks into the prediction.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluation


class SplitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(evaluation.SEED)
        # 300 rows, 6 labels, prevalence near the real dataset's marginals.
        marginals = np.array([0.20, 0.44, 0.05, 0.20, 0.25, 0.02])
        self.y = (rng.random((300, 6)) < marginals).astype(int)

    def test_split_sizes_and_disjointness(self):
        splits = evaluation.split_indices(self.y)
        self.assertEqual(len(splits["train"]), 210)
        self.assertEqual(len(splits["val"]), 45)
        self.assertEqual(len(splits["test"]), 45)
        combined = np.concatenate([splits["train"], splits["val"], splits["test"]])
        self.assertEqual(len(np.unique(combined)), 300)

    def test_split_is_deterministic(self):
        first = evaluation.split_indices(self.y)
        second = evaluation.split_indices(self.y)
        for name in ("train", "val", "test"):
            np.testing.assert_array_equal(first[name], second[name])

    def test_no_empty_split(self):
        splits = evaluation.split_indices(self.y)
        for name, idx in splits.items():
            self.assertGreater(len(idx), 0, name)


class MetricTests(unittest.TestCase):
    def test_emotions_metrics_names_are_explicit(self):
        y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        y_pred = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        result = evaluation.emotions_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["subset_accuracy"], 0.5)
        self.assertAlmostEqual(result["hamming_loss"], 1 / 12)
        self.assertEqual(set(result["per_label"]), set(evaluation.EMOTIONS))
        self.assertNotIn("accuracy", result)

    def test_all_zero_prediction_matches_majority_marginals(self):
        y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        y_pred = np.zeros_like(y_true)
        result = evaluation.emotions_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["per_label"]["Love"]["recall"], 0.0)
        self.assertAlmostEqual(result["per_label"]["Joy"]["f1"], 0.0)

    def test_per_label_accuracy_is_comparable_to_majority_baseline(self):
        """The reported per-label accuracy must use the same definition as the baseline."""
        y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        y_pred = np.zeros_like(y_true)
        model = evaluation.emotions_metrics(y_true, y_pred)
        baseline = evaluation.majority_baseline(y_true)
        # Identifying all-zero predictions on these rows is exactly the
        # baseline's own accuracy, so the two definitions must agree.
        self.assertAlmostEqual(
            model["per_label_accuracy"], baseline["per_label_majority_accuracy"]
        )

    def test_per_label_accuracy_differs_from_subset_accuracy(self):
        """Guards against relabelling one metric as the other."""
        y_true = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        y_pred = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]])
        result = evaluation.emotions_metrics(y_true, y_pred)
        # Neither row matches all six labels, so subset accuracy is 0 ...
        self.assertAlmostEqual(result["subset_accuracy"], 0.0)
        # ... while 10 of the 12 individual cells agree.
        self.assertAlmostEqual(result["per_label_accuracy"], 10 / 12)

    def test_topic_metrics_is_multiclass(self):
        names = ["A", "B", "C"]
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 2, 2, 2])
        result = evaluation.topic_metrics(y_true, y_pred, names)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertEqual(set(result["per_class"]), set(names))

    def test_majority_baseline_never_beats_itself(self):
        y_true = np.zeros((10, 6), dtype=int)
        baseline = evaluation.majority_baseline(y_true)
        self.assertAlmostEqual(baseline["per_label_majority_accuracy"], 1.0)
        self.assertAlmostEqual(baseline["hamming_loss"], 0.0)


class ThresholdTests(unittest.TestCase):
    def test_threshold_selection_separates_a_learnable_label(self):
        rng = np.random.default_rng(0)
        y_true = (rng.random((200, 6)) < 0.3).astype(int)
        scores = np.clip(y_true * 0.8 + rng.random((200, 6)) * 0.2, 0, 1)
        thresholds = evaluation.best_thresholds(y_true, scores)
        self.assertEqual(thresholds.shape, (6,))
        predictions = (scores >= thresholds).astype(int)
        self.assertGreater(predictions.mean(), 0.0)


class MaskingTests(unittest.TestCase):
    """Regression guard: padding must not change the pooled representation."""

    def _build(self, mask_zero):
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Attention,
            Bidirectional,
            Embedding,
            GlobalAveragePooling1D,
            Input,
            LSTM,
        )
        from tensorflow.keras.models import Model

        # Variable-length input lets us feed the same tokens with more padding.
        inputs = Input(shape=(None,))
        embedded = Embedding(50, 8, mask_zero=mask_zero)(inputs)
        lstm = Bidirectional(LSTM(4, return_sequences=True))(embedded)
        attended = Attention()([lstm, lstm])
        pooled = GlobalAveragePooling1D()(attended)
        return Model(inputs, pooled)

    def test_masked_model_is_padding_invariant(self):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(evaluation.SEED)
        model = self._build(mask_zero=True)
        tokens = [[3, 7, 11, 5]]
        short = tf.constant(tokens, dtype=tf.int32)
        padded = tf.constant([[0, 0, 3, 7, 11, 5]], dtype=tf.int32)
        np.testing.assert_allclose(
            model(short).numpy(), model(padded).numpy(), rtol=1e-5, atol=1e-5
        )

    def test_unmasked_model_is_not_padding_invariant(self):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(evaluation.SEED)
        model = self._build(mask_zero=False)
        short = tf.constant([[3, 7, 11, 5]], dtype=tf.int32)
        padded = tf.constant([[0, 0, 3, 7, 11, 5]], dtype=tf.int32)
        self.assertFalse(
            np.allclose(model(short).numpy(), model(padded).numpy(), rtol=1e-5, atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
