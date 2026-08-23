"""Learning algorithms: local, predictive, Hebbian, replay, continual."""

from learning.continual import ContinualGuard
from learning.hebbian import hebbian_update, oja_update, trace_update
from learning.nn import CfCCell, GRUCell, Linear, Module, Parameter, RMSNorm, Reservoir, Sequential, tanh, sigmoid
from learning.optim import Adam, SGD

__all__ = [
    "Adam",
    "CfCCell",
    "ContinualGuard",
    "GRUCell",
    "Linear",
    "Module",
    "Parameter",
    "RMSNorm",
    "Reservoir",
    "SGD",
    "Sequential",
    "hebbian_update",
    "oja_update",
    "sigmoid",
    "tanh",
    "trace_update",
]
