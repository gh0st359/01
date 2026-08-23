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

Selection rule: keep the substrate with lower late-window prediction MSE on the nursery sandbox, unless it is unstable. GRU is the integration default because it has trainable recurrent gates, BPTT-compatible local updates, and persistent hidden state at three timescales (1, 4, 16 ticks).

Workspace softmax is **not** a transformer language model. It scores seven competition features on a small set of candidates (capacity 7).
