# Deployment and profiles

Profiles in `shared/config.py`:

- `development` — small dims, fast tests
- `cpu` / `cloud` — default 15 GB class machines
- `single_gpu` / `multi_gpu` — larger dims if CUDA exists
- `high_throughput` — shorter imagination
- `long_running` — more archival + frequent checkpoints

`shared/hardware.py` picks a profile from RAM/CPU/GPU unless `O1_PROFILE` is set.

The organism never receives unrestricted internet, credentials, or physical actuators. Allowed effectors: simulator, research UI speech, project-local organism data directory.
