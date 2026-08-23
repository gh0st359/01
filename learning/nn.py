"""Inspectable numpy neural substrate.

Transformers are not the cognitive engine. Recurrent cells, closed-form
continuous-time cells, and reservoirs are first-class. Attention appears
only as a limited-capacity workspace competition primitive.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from shared.rng import RNG
from shared.types import Vector


def sigmoid(x: Vector) -> Vector:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def tanh(x: Vector) -> Vector:
    return np.tanh(x)


def relu(x: Vector) -> Vector:
    return np.maximum(x, 0.0)


def softmax(x: Vector) -> Vector:
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def cosine(a: Vector, b: Vector) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    aa, bb = a[:n], b[:n]
    na = float(np.linalg.norm(aa)) + 1e-12
    nb = float(np.linalg.norm(bb)) + 1e-12
    return float(np.dot(aa, bb) / (na * nb))


def l2_normalize(x: Vector) -> Vector:
    n = float(np.linalg.norm(x)) + 1e-12
    return x / n


class Parameter:
    def __init__(self, value: Vector, name: str) -> None:
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = np.zeros_like(self.value)
        self.name = name
        self.stable_shadow = self.value.copy()
        self.importance = np.zeros_like(self.value)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    def __init__(self, name: str) -> None:
        self.name = name
        self._params: list[Parameter] = []
        self._modules: list[Module] = []

    def register(self, param: Parameter) -> Parameter:
        self._params.append(param)
        return param

    def add(self, module: Module) -> Module:
        self._modules.append(module)
        return module

    def parameters(self) -> list[Parameter]:
        out = list(self._params)
        for module in self._modules:
            out.extend(module.parameters())
        return out

    def state_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for param in self.parameters():
            data[param.name] = {
                "value": param.value,
                "stable": param.stable_shadow,
                "importance": param.importance,
            }
        return data

    def load_state_dict(self, data: dict[str, Any]) -> None:
        lookup = {p.name: p for p in self.parameters()}
        for name, blob in data.items():
            if name in lookup:
                lookup[name].value = np.asarray(blob["value"], dtype=np.float64)
                lookup[name].stable_shadow = np.asarray(blob.get("stable", blob["value"]), dtype=np.float64)
                lookup[name].importance = np.asarray(blob.get("importance", np.zeros_like(blob["value"])), dtype=np.float64)

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()


def _xavier(rng: RNG, rows: int, cols: int) -> Vector:
    scale = np.sqrt(6.0 / (rows + cols))
    return rng.uniform(-scale, scale, (rows, cols))


class Linear(Module):
    def __init__(self, name: str, din: int, dout: int, rng: RNG, bias: bool = True) -> None:
        super().__init__(name)
        self.din = din
        self.dout = dout
        self.weight = self.register(Parameter(_xavier(rng, dout, din), f"{name}.w"))
        self.bias = self.register(Parameter(np.zeros(dout), f"{name}.b")) if bias else None

    def forward(self, x: Vector) -> Vector:
        y = self.weight.value @ x
        if self.bias is not None:
            y = y + self.bias.value
        return y

    def backward(self, x: Vector, grad_y: Vector, weight_decay: float = 0.0) -> Vector:
        self.weight.grad += np.outer(grad_y, x) + weight_decay * self.weight.value
        if self.bias is not None:
            self.bias.grad += grad_y
        return self.weight.value.T @ grad_y


class RMSNorm(Module):
    def __init__(self, name: str, dim: int) -> None:
        super().__init__(name)
        self.scale = self.register(Parameter(np.ones(dim), f"{name}.g"))

    def forward(self, x: Vector) -> Vector:
        rms = np.sqrt(np.mean(x * x) + 1e-6)
        return (x / rms) * self.scale.value


class GRUCell(Module):
    """Gated recurrent cell — persistent dynamical state."""

    def __init__(self, name: str, din: int, hidden: int, rng: RNG) -> None:
        super().__init__(name)
        self.din = din
        self.hidden = hidden
        self.wz = self.add(Linear(f"{name}.wz", din, hidden, rng))
        self.uz = self.add(Linear(f"{name}.uz", hidden, hidden, rng, bias=False))
        self.wr = self.add(Linear(f"{name}.wr", din, hidden, rng))
        self.ur = self.add(Linear(f"{name}.ur", hidden, hidden, rng, bias=False))
        self.wn = self.add(Linear(f"{name}.wn", din, hidden, rng))
        self.un = self.add(Linear(f"{name}.un", hidden, hidden, rng, bias=False))

    def forward(self, x: Vector, h: Vector) -> tuple[Vector, dict[str, Vector]]:
        z = sigmoid(self.wz.forward(x) + self.uz.forward(h))
        r = sigmoid(self.wr.forward(x) + self.ur.forward(h))
        n = tanh(self.wn.forward(x) + self.un.forward(r * h))
        h_next = (1.0 - z) * n + z * h
        cache = {"x": x, "h": h, "z": z, "r": r, "n": n, "h_next": h_next}
        return h_next, cache

    def backward(self, cache: dict[str, Vector], grad_h: Vector) -> tuple[Vector, Vector]:
        x, h, z, r, n = cache["x"], cache["h"], cache["z"], cache["r"], cache["n"]
        grad_n = grad_h * (1.0 - z)
        grad_z = grad_h * (h - n)
        grad_h_from_h = grad_h * z
        grad_n_pre = grad_n * (1.0 - n * n)
        grad_z_pre = grad_z * z * (1.0 - z)
        grad_x = np.zeros_like(x)
        grad_h_total = grad_h_from_h.copy()
        grad_x += self.wn.backward(x, grad_n_pre)
        grad_rh = self.un.backward(r * h, grad_n_pre)
        grad_r = grad_rh * h
        grad_h_total += grad_rh * r
        grad_r_pre = grad_r * r * (1.0 - r)
        grad_x += self.wr.backward(x, grad_r_pre)
        grad_h_total += self.ur.backward(h, grad_r_pre)
        grad_x += self.wz.backward(x, grad_z_pre)
        grad_h_total += self.uz.backward(h, grad_z_pre)
        return grad_x, grad_h_total


class CfCCell(Module):
    """Closed-form continuous-time cell inspired by Hasani et al. 2022.

    Used as a comparative substrate, not as a language model.
    x(t) ≈ σ(-f)·x(0) + (1-σ(-f))·g
    """

    def __init__(self, name: str, din: int, hidden: int, rng: RNG) -> None:
        super().__init__(name)
        self.ff = self.add(Linear(f"{name}.f", din + hidden, hidden, rng))
        self.gg = self.add(Linear(f"{name}.g", din + hidden, hidden, rng))
        self.time = self.add(Linear(f"{name}.t", din, hidden, rng))

    def forward(self, x: Vector, h: Vector, dt: float = 0.05) -> Vector:
        inp = np.concatenate([x, h])
        f = relu(self.ff.forward(inp))
        g = tanh(self.gg.forward(inp))
        tau = sigmoid(self.time.forward(x)) * (1.0 + 8.0 * dt)
        mix = sigmoid(-f * tau)
        return mix * h + (1.0 - mix) * g


class Reservoir(Module):
    """Echo-state reservoir with trainable readout."""

    def __init__(self, name: str, din: int, hidden: int, dout: int, rng: RNG) -> None:
        super().__init__(name)
        scale = 0.9 / np.sqrt(hidden)
        w = rng.normal((hidden, hidden), scale=scale)
        eig = np.max(np.abs(np.linalg.eigvals(w))) + 1e-6
        w = w * (0.9 / eig)
        self.w_res = self.register(Parameter(w, f"{name}.wres"))
        self.w_in = self.register(Parameter(rng.normal((hidden, din), scale=0.2), f"{name}.win"))
        self.readout = self.add(Linear(f"{name}.out", hidden, dout, rng))
        self.leak = 0.3

    def forward(self, x: Vector, h: Vector) -> tuple[Vector, Vector]:
        pre = tanh(self.w_in.value @ x + self.w_res.value @ h)
        h_next = (1.0 - self.leak) * h + self.leak * pre
        return self.readout.forward(h_next), h_next


class Sequential(Module):
    def __init__(self, name: str, layers: Iterable[Module]) -> None:
        super().__init__(name)
        self.layers = [self.add(layer) for layer in layers]


class Encoder(Module):
    def __init__(self, name: str, din: int, hidden: int, dout: int, rng: RNG) -> None:
        super().__init__(name)
        self.l1 = self.add(Linear(f"{name}.l1", din, hidden, rng))
        self.l2 = self.add(Linear(f"{name}.l2", hidden, dout, rng))

    def forward(self, x: Vector) -> Vector:
        return tanh(self.l2.forward(relu(self.l1.forward(x))))


class Decoder(Module):
    def __init__(self, name: str, din: int, hidden: int, dout: int, rng: RNG) -> None:
        super().__init__(name)
        self.l1 = self.add(Linear(f"{name}.l1", din, hidden, rng))
        self.l2 = self.add(Linear(f"{name}.l2", hidden, dout, rng))

    def forward(self, x: Vector) -> Vector:
        return self.l2.forward(relu(self.l1.forward(x)))


class FamiliarityNet(Module):
    """Learns to reconstruct inputs; reconstruction error ≈ novelty."""

    def __init__(self, name: str, dim: int, rng: RNG) -> None:
        super().__init__(name)
        hidden = max(16, dim // 2)
        self.enc = self.add(Encoder(f"{name}.e", dim, hidden, hidden, rng))
        self.dec = self.add(Decoder(f"{name}.d", hidden, hidden, dim, rng))

    def novelty(self, x: Vector) -> float:
        recon = self.dec.forward(self.enc.forward(x))
        return float(np.mean((recon - x) ** 2))

    def train_step(self, x: Vector, lr: float) -> float:
        h = self.enc.forward(x)
        recon = self.dec.forward(h)
        err = recon - x
        loss = float(np.mean(err ** 2))
        grad_recon = (2.0 / err.size) * err
        # local linearization through relu/tanh via straight-through-ish updates
        self.dec.l2.backward(relu(self.dec.l1.forward(h)), grad_recon)
        grad_h = self.dec.l1.backward(h, (self.dec.l2.weight.value.T @ grad_recon) * (self.dec.l1.forward(h) > 0))
        self.enc.l2.backward(relu(self.enc.l1.forward(x)), grad_h * (1.0 - np.tanh(self.enc.l2.forward(relu(self.enc.l1.forward(x)))) ** 2))
        for param in self.parameters():
            param.value -= lr * np.clip(param.grad, -1.0, 1.0)
            param.zero_grad()
        return loss
