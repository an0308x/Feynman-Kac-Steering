# FK Steering — Feynman–Kac steering for protein diffusion

A PyTorch implementation of inference-time steering for diffusion-based protein design,
reproducing the method from:

> **Controllable protein design through Feynman–Kac steering**  
> Erik Hartman, Jonas Wallin, Johan Malmström, Jimmy Olsson  
> arXiv:2511.09216 (2025)

---

## Overview

Diffusion models generate realistic protein structures but offer little control over
biochemical properties like binding affinity or secondary structure composition. FK
steering solves this by wrapping any trained diffusion model in a
**Sequential Monte Carlo (SMC) loop** that re-weights and resamples a particle ensemble
at each denoising step according to user-defined reward functions — **without any
gradient computation or model retraining**.

```
Gaussian noise  xT
       │
   ┌───▼──────────────────────────────────┐
   │  RFdiffusion  pθ(xt-1 | xt)         │  ← frozen
   └───┬──────────────────────────────────┘
       │  xt  (N particles)
   ┌───▼──────────────────────────────────┐
   │  FK steering                          │
   │  1. predict  x̂0|t  via denoiser      │
   │  2. ProteinMPNN → sequence st         │
   │  3. PyRosetta pack + relax → x̃0|t    │
   │  4. reward  r(x̃0|t, st)              │
   │  5. potential  Gt                     │
   │  6. systematic resample               │
   └───┬──────────────────────────────────┘
       │  xt-1 … x0  (steered designs)
```

---

## Repository structure

```
fk_steering/
├── fk_steering/
│   ├── __init__.py
│   ├── steering.py            # FK steering engine + potential functions
│   ├── rewards.py             # ΔG, charge, secondary-structure rewards
│   └── rfdiffusion_adapter.py # thin wrapper around RFdiffusion
├── scripts/
│   └── fk_binder_design.py    # command-line design campaign runner
├── configs/
│   └── default_binder.yaml    # optimal hyperparameters from the paper
├── tests/
│   └── test_fk_steering.py    # unit + integration tests (no external deps)
├── pyproject.toml
└── README.md
```

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/fk-steering.git
cd fk-steering
pip install -e ".[dev]"
```

### 2. Install RFdiffusion (required for real protein design)

```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion
pip install -e .
export RFDIFFUSION_PATH=$(pwd)
```

Download weights:
```bash
mkdir -p models
wget -P models https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc6/Complex_base_ckpt.pt
wget -P models https://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
```

### 3. Install ProteinMPNN

```bash
git clone https://github.com/dauparas/ProteinMPNN.git
export PROTEIN_MPNN_PATH=$(pwd)/ProteinMPNN
```

### 4. Install PyRosetta

PyRosetta requires a [free academic licence](https://www.pyrosetta.org/downloads).
Follow the installer instructions on the PyRosetta website.

---

## Quick start

### Run with optimal paper parameters (streptolysin O)

```bash
python scripts/fk_binder_design.py \
    --target   targets/4HSC.pdb      \
    --hotspots A110 A115 A117        \
    --binder_length 24               \
    --n_particles 50                 \
    --potential immediate            \
    --temperature 10.0               \
    --resample_interval 2            \
    --guidance_start 20              \
    --reward binding                 \
    --output results/SLO/
```

### Steer toward negative charge

```bash
python scripts/fk_binder_design.py \
    --target targets/4HSC.pdb \
    --reward charge           \
    --target_charge -5.0      \
    --output results/charged/
```

### Steer toward α-helix secondary structure

```bash
python scripts/fk_binder_design.py \
    --target targets/4HSC.pdb \
    --reward secondary        \
    --target_helix 1.0        \
    --output results/helical/
```

### Programmatic usage

```python
from fk_steering import (
    FKSteering, FKSteeringConfig,
    BindingEnergyReward, SequenceStructurePipeline,
    RFdiffusionAdapter,
)

# Model and reward
model    = RFdiffusionAdapter(rfdiffusion_path="/path/to/RFdiffusion")
pipeline = SequenceStructurePipeline()
reward   = BindingEnergyReward(pipeline=pipeline)

# Steering configuration (optimal from paper)
config = FKSteeringConfig(
    n_particles       = 50,
    temperature       = 10.0,
    resample_interval = 2,
    guidance_start    = 20,
    potential         = "immediate",
    n_reward_samples  = 5,
)

steering = FKSteering(config=config, reward_fn=reward, diffusion_model=model)

context = {
    "target_pdb":      "targets/4HSC.pdb",
    "hotspot_residues": [{"chain": "A", "resnum": r} for r in [110, 115, 117]],
    "binder_length":   24,
}

particles = steering.run(context, n_timesteps=50)
# particles[0] is the top-ranked design
print(f"Top reward: {particles[0].reward:.2f}")
```

---

## Potential functions

| Potential | Domain | Functional form | Notes |
|-----------|--------|-----------------|-------|
| `immediate` | xₜ | exp(rₜ) | Exact FK; highest structural concordance (Fig. 5) |
| `difference` | (xₜ, xₜ₊₁) | exp(rₜ − rₜ₊₁) | Preserves diversity best (Fig. 3b) |
| `max` | xₜ:T | exp(max_{s≥t} rₛ) | Tracks running maximum |
| `sum` | xₜ:T | exp(∑_{s=t}^T rₛ) | Strongest guidance, most diversity collapse |

---

## Hyperparameter guide (from Fig. 3d)

| Parameter | Optimal | Effect of increase |
|-----------|---------|-------------------|
| `n_particles` | 50 | ↑ reward and diversity |
| `temperature τ` | 10 | ↓ τ → ↑ reward, ↓ diversity |
| `guidance_start t_start` | 20 | Delay improves reward (early proxies unreliable) |
| `resample_interval Δt` | 2 | Larger Δt preserves diversity at cost of control |
| `n_reward_samples` | 5 | ↑ reward, no diversity cost (Fig. 4) |

---

## Running the tests

No external dependencies needed for the test suite:

```bash
pytest tests/ -v
```

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{hartman2025fksteering,
  title   = {Controllable protein design through {Feynman--Kac} steering},
  author  = {Hartman, Erik and Wallin, Jonas and Malmstr{\"o}m, Johan and Olsson, Jimmy},
  journal = {arXiv preprint arXiv:2511.09216},
  year    = {2025},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
