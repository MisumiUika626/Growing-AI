import numpy as np

from .autograd.tensor import Tensor


def accuracy(logits, targets):
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")

    if logits.data.ndim != 2 or 0 in logits.data.shape:
        raise ValueError("logits must have non-empty shape (B, C)")
    targets = np.asarray(targets)
    if targets.shape != (logits.data.shape[0],):
        raise ValueError("targets must have shape (B,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("targets must contain integer class indices")
    if np.any(targets < 0) or np.any(targets >= logits.data.shape[1]):
        raise ValueError("targets must satisfy 0 <= target < C")
    targets = targets.astype(np.int64, copy=False)
    predictions = np.argmax(logits.data, axis=1)
    correct = predictions == targets
    correct_count = np.sum(correct)
    total_count = len(correct)
    accuracy_value = correct_count / total_count
    return float(accuracy_value)
