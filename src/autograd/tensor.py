import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def _unbroadcast_grad(self, grad, target_shape):
        if grad.shape == target_shape:
            return grad
        while grad.ndim > len(target_shape):
            grad = np.sum(grad, axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += self._unbroadcast_grad(out.grad, self.data.shape)
            other.grad += self._unbroadcast_grad(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, (self,), "-")

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += self._unbroadcast_grad(out.grad * other.data, self.data.shape)
            other.grad += self._unbroadcast_grad(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    # only 2d matrix supported
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if not self.data.ndim == 2 or not other.data.ndim == 2:
            raise ValueError("Both tensors must be 2D for matrix multiplication")
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += np.exp(self.data) * out.grad

        out._backward = _backward
        return out

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("log is only defined for positive values")
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += 1 / self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("exponent must be an int or float")
        out = Tensor(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            if exponent == 0:
                return  # x**0 is constant, including at x=0.
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self * other**-1

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other * self**-1

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad

            if axis != None and keepdims == False:
                grad = np.expand_dims(grad, axis=axis)

            self.grad += grad * np.ones_like(self.data)

        out._backward = _backward
        return out

    def sigmoid(self):
        # Both branches exponentiate non-positive values to avoid overflow.
        positive = self.data >= 0
        values = np.empty_like(self.data)
        values[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        values[~positive] = exp_x / (1.0 + exp_x)
        out = Tensor(values, (self,), "sigmoid")

        def _backward():
            self.grad += out.grad * out.data * (1.0 - out.data)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise TypeError("alpha must be an int or float")
        if not 0 < alpha < 1:
            raise ValueError("alpha must satisfy 0 <= alpha < 1")
        out = Tensor(
            np.where(self.data > 0, self.data, alpha * self.data), (self,), "leaky_relu"
        )

        def _backward():
            self.grad += out.grad * np.where(self.data > 0, 1, alpha)

        out._backward = _backward
        return out

    def backward(self):
        if self.data.shape != ():
            raise RuntimeError("backward() can only be called on a scalar Tensor")
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        for node in topo:
            node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def mean(self):
        out = Tensor(np.mean(self.data), (self,), "mean")

        def _backward():
            self.grad += out.grad * np.ones_like(self.data) / self.data.size

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape})"
