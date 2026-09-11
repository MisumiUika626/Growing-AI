"""Regression checks for issues found in the final project audit."""

import contextlib
import io
import unittest

import numpy as np

from src.autograd.engine import Value
from src.autograd.tensor import Tensor
from src.experiments import build_toy_experiment
from src.metrics import accuracy
from src.nn.activations import Sigmoid
from src.nn.losses import CrossEntropyLoss
from src.trainers.trainer import ToyTrainer


class TestFinalization(unittest.TestCase):
    def test_zero_power_has_zero_gradient(self):
        for cls in (Value, Tensor):
            for value in (0.0, -2.0, 3.0):
                with self.subTest(cls=cls, value=value), np.errstate(all="raise"):
                    x = cls(value)
                    constant = x**0
                    constant.backward()
                    self.assertEqual(float(constant.data), 1.0)
                    self.assertEqual(float(x.grad), 0.0)
                    (constant + 2 * x).backward()
                    self.assertEqual(float(x.grad), 2.0)

    def test_value_backward_clears_intermediate_gradients(self):
        x = Value(2.0)
        y = x * x
        loss = y * y
        for reset_leaf in (False, False, True):
            if reset_leaf:
                x.grad = 0.0
            loss.backward()
            self.assertEqual(x.grad, 32.0)
            self.assertEqual(y.grad, 8.0)

    def test_sigmoid_extreme_and_scalar_inputs(self):
        for data in ([-1000.0, -2.0, 0.0, 2.0, 1000.0], -1000.0, 0.0, 1000.0):
            with self.subTest(data=data):
                x = Tensor(data)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    result = Sigmoid().forward(x)
                    (3 * result + x).sum().backward()
                self.assertTrue(np.isfinite(result.data).all())
                self.assertTrue(np.isfinite(x.grad).all())
                if isinstance(data, list):
                    expected = np.array([
                        0.0, 0.11920292202211755, 0.5,
                        0.8807970779778823, 1.0,
                    ])
                else:
                    expected = np.array(
                        0.0 if data < 0 else 1.0 if data > 0 else 0.5
                    )
                np.testing.assert_allclose(result.data, expected)
                np.testing.assert_allclose(x.grad, 1 + 3 * expected * (1 - expected))

    def test_classification_rejects_invalid_targets(self):
        logits = Tensor([[3.0, 0.0], [0.0, 3.0]])
        invalid = [
            ([[0], [1]], ValueError), ([0], ValueError), (0, ValueError),
            ([-1, 1], ValueError), ([0, 2], ValueError),
            ([0.9, 1.9], TypeError), ([0.0, 1.0], TypeError),
            ([True, False], TypeError), (["0", "1"], TypeError),
        ]
        for fn in (accuracy, CrossEntropyLoss().forward):
            for targets, error in invalid:
                with self.subTest(fn=fn, targets=targets), self.assertRaises(error):
                    fn(logits, targets)

    def test_classification_rejects_empty_or_non_matrix_logits(self):
        for fn in (accuracy, CrossEntropyLoss().forward):
            for shape in ((2,), (0, 2), (2, 0), (1, 2, 2)):
                with self.subTest(fn=fn, shape=shape), self.assertRaises(ValueError):
                    fn(Tensor(np.zeros(shape)), np.array([], dtype=int))

    def test_classification_accepts_integer_dtypes(self):
        logits = Tensor([[3.0, 0.0], [0.0, 3.0]])
        for dtype in (np.int32, np.int64, np.uint8, np.uint64):
            targets = np.array([0, 1], dtype=dtype)
            self.assertEqual(accuracy(logits, targets), 1.0)
            self.assertAlmostEqual(float(CrossEntropyLoss().forward(logits, targets).data), 0.04858735157374196)

    def test_linear_dataset_defaults_train_without_divergence(self):
        for model_name in ("linear", "mlp"):
            with self.subTest(model=model_name):
                experiment = build_toy_experiment("linear", model_name)
                with contextlib.redirect_stdout(io.StringIO()), np.errstate(over="raise", invalid="raise", divide="raise"):
                    history = ToyTrainer().fit(
                        experiment["model"], experiment["dataset"],
                        experiment["optimizer"], experiment["epochs"],
                    )
                self.assertTrue(np.isfinite(history).all())
                self.assertLess(history[-1], history[0])
                for parameter in experiment["model"].parameters():
                    self.assertTrue(np.isfinite(parameter.data).all())


if __name__ == "__main__":
    unittest.main()
