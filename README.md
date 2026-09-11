# BloomingTensor

A NumPy project I built while learning deep learning. It implements automatic differentiation, fully connected networks, and SGD to help me understand how gradients are computed and how neural networks learn.

## Features

- Autograd with arithmetic, broadcasting, 2D matrix multiplication, sums, and means.
- Linear layers, MLPs, activation functions, MSE, cross-entropy, and SGD.
- Linear and nonlinear regression, XOR, Two Moons, and MNIST experiments.

## Getting started

Requires Python 3.9+. Run these Bash commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Run a small experiment:

```bash
python3 -m src.main --dataset xor --model mlp
python3 -m src.main --dataset twomoon --model mlp
```

Other datasets are `linear` and `nonlinear`. Choose `linear` or `mlp` as the model. Loss curves are saved to `plots/`; XOR and Two Moons also produce prediction-surface plots.

### MNIST

Download the four data files and leave them gzip-compressed:

```bash
mkdir -p data/mnist/raw
for file in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    curl -fL "https://storage.googleapis.com/cvdf-datasets/mnist/$file" \
        -o "data/mnist/raw/$file"
done
```

Check the training pipeline with a small subset:

```bash
python3 -m src.main_mnist --epochs 2 --train-limit 512 --test-limit 256
```

Run `python3 -m src.main_mnist` to train on the full dataset with the default configuration. It reports training loss, test loss, and test accuracy. Use `--data-dir` if your files are stored elsewhere.

## Example

```python
from src.autograd.tensor import Tensor

x = Tensor([1.0, 2.0, 3.0])
loss = (x * x).sum()
loss.backward()

print(x.grad)  # [2. 4. 6.]
```

![MLP on Two Moons](docs/assets/twomoon_mlp_prediction_surface.png)

An MLP fitted to the Two Moons training set using MSE. Colors show raw model outputs; the black line marks an output of `0.5`.

## Notes

This is a learning project. Matrix multiplication supports only 2D tensors, and `backward()` requires a scalar output. Each backward call recomputes gradients rather than accumulating across calls.

The small-dataset experiments evaluate training data only; XOR and Two Moons outputs are not probabilities. MNIST uses the standard test set without a separate validation set.

See the [experiment notes](docs/experiments.md) and [architecture overview](docs/architecture.md) for more details.
