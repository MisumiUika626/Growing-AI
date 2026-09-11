# BloomingTensor Architecture

## Principles

- Prefer simple, understandable designs with clear responsibilities.
- Separate numerical operations, model composition, training, and parameter updates.
- Preserve learning value; avoid premature abstraction and hypothetical systems.

## Shared core and separate learning tracks

The project has one reusable numerical/model core and two explicit experiment tracks. `MLP` is shared deliberately: the difference between the tracks is the data contract, loss, trainer, and evaluation—not a second MLP implementation.

| Responsibility | Toy foundations | MNIST classification |
| --- | --- | --- |
| Input | Individual small vectors | Mini-batches `(B,784)` |
| Output | Usually `(1,1)` raw scalar | `(B,10)` logits |
| Target | Scalar Tensor | Integer labels `(B,)` |
| Loss | MSE | Stable CrossEntropy |
| Evaluation | Training loss / visual inspection | Held-out test loss and accuracy |
| Entry | `src.main` | `src.main_mnist` |

## Current architecture snapshot

```text
src/
├── autograd/
│   ├── engine.py       # Scalar Value automatic differentiation
│   └── tensor.py       # NumPy Tensor operations and reverse-mode gradients
├── nn/                 # Linear, activation, and Sequential components
├── models/mlp.py       # MLP construction from reusable nn components
├── datasets/           # Toy datasets, MNIST IDX parser, and DataLoader
├── configs/            # Toy settings plus an explicit MNIST configuration
├── optimizers/sgd.py   # Parameter update and gradient reset
├── trainers/
│   ├── trainer.py                # ToyTrainer: per-sample MSE
│   └── classification_trainer.py # Mini-batch CrossEntropy + test evaluation
├── experiments.py      # Toy dataset/model/config assembly only
├── visualization.py    # Two-dimensional toy raw-output surfaces
├── main.py             # Toy experiment CLI
├── main_mnist.py       # MNIST classification CLI
└── show_mnist.py       # Save one MNIST image without training

tests/                  # Autograd, layer, model, and loss tests
docs/                   # Architecture, experiments, roadmap, and notes
```

## Toy data flow

```text
dataset name + model name
            ↓
build_toy_experiment(dataset_name, model_name)
            ↓
Dataset + Model + SGD + configuration
            ↓
zero_grad → forward → MSE → backward → SGD step
            ↓
fixed-parameter training-set evaluation → loss history → plots
```

`build_toy_experiment(dataset_name, model_name)` first selects the toy dataset and its matching configuration, seeds NumPy, then constructs either `Linear` or the shared `MLP`. `ToyTrainer` preserves the original scalar-output MSE path as a regression bench.

## MNIST data flow

```text
train/test IDX gzip files
            ↓
Dataset: images (N,784), labels (N,)
            ↓
DataLoader: shuffled train batches / ordered test batches
            ↓
shared MLP [784,128,10]
            ↓
stable CrossEntropy → backward → SGD step
            ↓
fixed-parameter test loss + argmax accuracy
```

The classification path does not call `Softmax` inside `CrossEntropyLoss`; it computes stable log probabilities directly. `Softmax` remains available when visible class probabilities are needed.

## Responsibility boundaries

- `Tensor` owns numerical values, graph edges, local derivative closures, topological traversal, and gradient storage.
- `Linear`, activations, `Sequential`, and `MLP` organize Tensor operations and expose trainable parameters.
- Toy Dataset objects expose individual samples. The MNIST Dataset stores aligned image/label arrays and also supports indexed samples.
- `DataLoader` owns batch boundaries and reproducible shuffling; it does not train models.
- `ToyTrainer` owns the original per-sample MSE sequence.
- `ClassificationTrainer` owns mini-batch CrossEntropy training and held-out evaluation.
- `SGD` only clears gradients and updates parameter data.
- `visualization.py` is restricted to two-dimensional toy raw-output surfaces.
- `show_mnist.py` only reconstructs a `(784,)` sample as `(28,28)` and saves it.
- `main.py` and `main_mnist.py` are separate composition roots for their respective tracks.

The two trainers remain separate because their target shapes, losses, batching, and evaluation contracts differ materially. Shared behavior should only be extracted after repetition becomes both stable and useful.

## Evolution policy

This structure is a snapshot of the current learning stage, not a permanent contract. Modules should be added, merged, renamed, or removed only when a verified learning step changes their responsibilities. Keep this document aligned with significant structural changes without designing systems before they are implemented.
