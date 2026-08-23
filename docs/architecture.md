# Architecture

```
world  →  sensorium  →  perception  →  entities / spatial / temporal
                ↓
        predictive world model  →  prediction error
                ↓
 specialized candidates  →  global workspace (capacity 7)  →  broadcast
                ↓
        recurrent core x(t)  →  x(t+Δt)
         ↙        ↓         ↘
   action     language     memory
```

Language is an I/O modality. It is not the mind.

## Persistent dynamics

Each runtime tick, regardless of human input:

1. Sense the simulated body and world.
2. Encode hierarchical percepts (pixels → features → proto-objects).
3. Track entities (identity + permanence).
4. Predict the next latent; compute error.
5. Form workspace candidates (vision, surprise, recall, language, goals, imagined futures, contradictions).
6. Compete; broadcast winners.
7. Update the recurrent core and homeostatic / intrinsic variables.
8. Form or decay goals; imagine a few action futures; decide.
9. Act through `MotorCommand`.
10. Encode an episode when surprise or schedule warrants it.
11. Possibly speak if a communicative intent exceeds threshold **and** grounded words exist.
12. Enter rest / consolidation / simulation / maintenance when internal variables require it.

## Separation of organism and instrumentation

`Organism.observe_state()` is a researcher window. Hidden probes and telemetry SQLite are not part of the self-model. Introspective speech can only mention states the organism actually represents and has words for.

## Reality bridge

`SensorFrame`, `MotorCommand`, `BodyState`, `AudioFrame`, `TactileFrame`, `WorldObservation` are the only world interface. Physical actuators are disabled.
