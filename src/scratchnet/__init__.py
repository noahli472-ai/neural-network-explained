"""scratchnet: a small neural-network library written from scratch in NumPy.

Everything the website shows (a single neuron, small networks learning 2-D patterns,
a handwritten-digit classifier) is trained with this package.
"""

from .layers import Dense, Layer, ReLU, Sigmoid, Tanh
from .losses import BinaryCrossEntropy, SoftmaxCrossEntropy, softmax
from .network import Sequential, mlp
from .optim import SGD, Adam
from .training import accuracy, fit

__all__ = [
    "SGD",
    "Adam",
    "BinaryCrossEntropy",
    "Dense",
    "Layer",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "SoftmaxCrossEntropy",
    "Tanh",
    "accuracy",
    "fit",
    "mlp",
    "softmax",
]

__version__ = "0.1.0"
