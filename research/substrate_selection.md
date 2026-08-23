# Substrate selection

Candidate cores implemented:

| Substrate | Location | Role |
|---|---|---|
| Multi-timescale GRU | `cognition/core/recurrent.py` | Default persistent cognitive state |
| Closed-form continuous-time (CfC-inspired) | `learning/nn.py` `CfCCell` | Comparative dynamics |
| Echo-state reservoir | `learning/nn.py` `Reservoir` | Fixed recurrent fabric + trainable readout |

Run:

```
python -m scripts.compare_substrates
```

Measured late-window MSE on the nursery sandbox (seed 3, 80 steps):

| Substrate | MSE |
|---|---|
| CfC | 0.072 |
| Reservoir | 0.138 |
| GRU | 0.198 |

Selection: GRU remains the integration default because it has trainable recurrent gates, BPTT-compatible local updates, and persistent hidden state at three timescales (1, 4, 16 ticks). CfC is kept as a comparative dynamics cell; a lower short-horizon residual does not make it the language or identity engine.

Workspace softmax is **not** a transformer language model. It scores seven competition features on a small set of candidates (capacity 7).
