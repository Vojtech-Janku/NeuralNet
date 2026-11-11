# Neural Network from Scratch (C++17)

A small deep-learning library written from scratch in C++17: tensors, dense and convolutional layers,
backpropagation, SGD / Momentum / Adam, softmax with cross-entropy, trained and evaluated on
[Fashion-MNIST](https://arxiv.org/abs/1708.07747). Only the C++ standard library is used; no
linear-algebra or ML libraries.

It started as a university semester project: an MLP from scratch, with even `<algorithm>` forbidden,
and a target of at least 88% test accuracy within 30 minutes of training. I'm now extending it with new
layer types, optimizers, tests and cleaner architecture.

## Results

Fashion-MNIST (60,000 training / 10,000 test images, 10 classes), 20 epochs, single-threaded:

| Model | Architecture | Train accuracy | Test accuracy | Training time* |
|---|---|---|---|---|
| MLP | 784 → 64 (ReLU) → 30 (ReLU) → 10 (Softmax) | 90.9% | **88.6%** | ~2 min |
| CNN | 1 conv filter 5×5 (ReLU) → 10 (Sigmoid) | 82.8% | **82.9%** | ~1.5 min |

Both use mini-batches of 64, Momentum (0.9) and learning-rate decay. Training is deterministic (fixed
random seed), so these numbers are reproducible.

\* Measured on a 12-thread desktop CPU running several jobs at once; the code itself is single-threaded.

## Features

- **Layers**: dense, 2D convolution
- **Activations**: Step, ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax (with cross-entropy gradient)
- **Optimizers**: SGD, Momentum, Adam, learning-rate decay
- **Initialization**: He (Gaussian) or uniform, reproducible from one seeded generator
- **Tested**: GoogleTest suite with finite-difference gradient checks for every activation and for
  softmax + cross-entropy, plus forward-pass and training tests

## Quick start

Requirements: a C++17 compiler (tested with GCC 13), CMake ≥ 3.20, and Python 3 with the
[`kaggle`](https://github.com/Kaggle/kaggle-api) package and a Kaggle API token for the data download.

```bash
python3 data/download_data.py   # download Fashion-MNIST and split it into data/*.csv
bash run.sh                     # build (CMake, Release) and train + evaluate the MLP
ctest --test-dir build          # run the tests
```

The program prints the train and test accuracy, and writes its predictions to `data/`. The experiments
(the MLP and the CNN, with their hyperparameters) are defined in [`apps/main.cpp`](apps/main.cpp).

## Known limitations

- The conv layer has one input channel and one kernel, no stride or padding, and must be the first layer.
- Samples are processed one at a time, and the data isn't shuffled between epochs.
- Multithreading (OpenMP) is off by default: the current pragmas make small layers slower, not faster.

## Roadmap

- **Layers**: multi-channel convolution, max pooling, dropout, batch normalization; freely stackable layers
- **Performance**: batched training, cache-blocked and SIMD matrix multiplication, working OpenMP, benchmarks
- **Usability**: command-line interface for experiments and hyperparameters, model save/load, CI
