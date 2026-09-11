# BloomingTensor Roadmap

## Project Direction

Learn AI engineering by implementing core components and connecting them into
small, understandable systems: software engineering → deep learning →
Transformers → agents → reinforcement learning.

## Completed

- Scalar automatic differentiation for basic arithmetic, shared-node gradient accumulation, and reverse topological propagation.
- NumPy-backed Tensor automatic differentiation for arithmetic, broadcasting, two-dimensional matrix multiplication, reductions, `exp`, `log`, ReLU, and Leaky ReLU.
- Linear, Sequential, and MLP components with discoverable parameters.
- Numerically stable Sigmoid, MSE training, per-sample SGD, and fixed-parameter epoch evaluation.
- Reproducible linear, `y = x^2`, XOR, and Two Moons datasets with independently selected Linear/MLP models.
- A command-line experiment runner that saves timestamped loss curves and two-dimensional prediction surfaces for XOR and Two Moons.
- Axis-aware Tensor reduction, stable row-wise Softmax, stable CrossEntropy, integer-label accuracy, and batch loading.
- A separate MNIST IDX loader, `[784,128,10]` MLP runner, classification trainer, held-out test evaluation, and sample visualization.
- Unit tests for current autograd, layer, model, activation, loss, metric, and batch behavior.

## Current Focus

Keep the original toy MSE experiments and the MNIST classification track explicitly separated while adding visible, reproducible evidence to the MNIST path. The shared `Tensor`, `MLP`, and `SGD` core should remain common to both.

## Next

1. Persist focused forward/backward tests for axis-aware reductions, including restored gradient shapes when `keepdims=False`.
2. Add MNIST train/test loss curves, test-accuracy curves, prediction grids, and a confusion matrix.
3. Run and record a bounded full-data MNIST baseline with its configuration and honest limitations.
4. Add a binary-classification contract with BCE, explicit threshold accuracy, and a held-out split before presenting XOR or Two Moons as evaluated classifiers.
5. Add centered finite-difference gradient checks for representative Tensor operations and layer parameters.

Transformer, agent, and reinforcement-learning work remain long-term learning directions, not current repository capabilities.
