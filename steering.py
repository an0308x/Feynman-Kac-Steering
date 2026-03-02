"""
Feynman-Kac Steering for Diffusion-Based Protein Design
========================================================
Implementation based on:
    Hartman et al. (2025) "Controllable protein design through Feynman-Kac steering"
    arXiv:2511.09216

This module wraps RFdiffusion's denoising loop with a particle-based Sequential
Monte Carlo (SMC) steering layer.  At every guidance step the particle ensemble is
re-weighted by user-defined potential functions and resampled (with replacement),
duplicating high-reward trajectories and pruning low-reward ones without any
gradient computation or model fine-tuning.

Architecture overview
---------------------
                 Gaussian noise  xT
                        │
          ┌─────────────▼──────────────────┐
          │  RFdiffusion  pθ(xt-1 | xt)    │  ← frozen, never modified
          └─────────────┬──────────────────┘
                        │  xt  (N particles)
          ┌─────────────▼──────────────────┐
          │  FK steering layer              │
          │   1. denoised proxy  x̂0|t      │
          │   2. ProteinMPNN → sequence st  │
          │   3. pack + relax  → x̃0|t      │
          │   4. reward  r(x̃0|t, st)       │
          │   5. potential  Gt              │
          │   6. resample particles         │
          └─────────────┬──────────────────┘
                        │  xt-1
                       ...
                        │  x0  (steered designs)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Potential functions  (Table 1 in the paper)
# ---------------------------------------------------------------------------

class PotentialFunction:
    """Base class for FK steering potentials."""

    def __call__(self, rewards: torch.Tensor, rewards_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class ImmediatePotential(PotentialFunction):
    """G_t = exp(r_t).  Reacts immediately to the current reward signal."""

    def __call__(self, rewards: torch.Tensor, rewards_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.exp(rewards)


class DifferencePotential(PotentialFunction):
    """G_t = exp(r_t - r_{t+1}).  Exact FK decomposition — differences across steps."""

    def __call__(self, rewards: torch.Tensor, rewards_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if rewards_prev is None:
            # Boundary: G_T = 1
            return torch.ones_like(rewards)
        return torch.exp(rewards - rewards_prev)


class MaxPotential(PotentialFunction):
    """G_t = exp(max_{s>=t} r_s).  Tracks the running maximum reward."""

    def __init__(self) -> None:
        self._running_max: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._running_max = None

    def __call__(self, rewards: torch.Tensor, rewards_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._running_max is None:
            self._running_max = rewards.clone()
        else:
            self._running_max = torch.maximum(self._running_max, rewards)
        return torch.exp(self._running_max)


class SumPotential(PotentialFunction):
    """G_t = exp(sum_{s=t}^T r_s).  Accumulates reward over the trajectory."""

    def __init__(self) -> None:
        self._cumsum: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._cumsum = None

    def __call__(self, rewards: torch.Tensor, rewards_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._cumsum is None:
            self._cumsum = rewards.clone()
        else:
            self._cumsum = self._cumsum + rewards
        return torch.exp(self._cumsum)


POTENTIAL_REGISTRY: dict[str, type] = {
    "immediate": ImmediatePotential,
    "difference": DifferencePotential,
    "max": MaxPotential,
    "sum": SumPotential,
}


def build_potential(name: str) -> PotentialFunction:
    name = name.lower()
    if name not in POTENTIAL_REGISTRY:
        raise ValueError(f"Unknown potential '{name}'. Choose from {list(POTENTIAL_REGISTRY)}")
    return POTENTIAL_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Particle state
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    """One trajectory in the FK ensemble."""
    xt: torch.Tensor              # current noisy structure  (shape determined by RFdiffusion)
    reward: Optional[float] = None
    log_weight: float = 0.0
    history: List[float] = field(default_factory=list)   # reward at each guided step

    def update_reward(self, r: float) -> None:
        self.reward = r
        self.history.append(r)


# ---------------------------------------------------------------------------
# Core FK steering class
# ---------------------------------------------------------------------------

@dataclass
class FKSteeringConfig:
    """Hyperparameters for FK steering (see Table 2 / Fig. 3d in paper)."""
    n_particles: int = 50
    """Number of parallel diffusion trajectories."""

    temperature: float = 10.0
    """τ — controls sharpness of resampling.  Lower τ = stronger guidance, less diversity."""

    resample_interval: int = 2
    """Δt — resample every this many denoising steps."""

    guidance_start: int = 20
    """t_start — begin steering at this timestep (counting from 0 = x0).
    Guidance is delayed because early denoised proxies are unreliable."""

    potential: str = "immediate"
    """Potential function name: 'immediate' | 'difference' | 'max' | 'sum'."""

    n_reward_samples: int = 5
    """n — number of sequence/relaxation samples used to average the reward (Fig. 4)."""

    reward_aggregation: str = "mean"
    """How to aggregate multiple reward samples: 'mean' or 'max'."""

    seed: Optional[int] = None


class FKSteering:
    """
    Inference-time FK steering wrapper for diffusion-based protein generation.

    Usage
    -----
    >>> steering = FKSteering(config, reward_fn, diffusion_model)
    >>> designs = steering.run(context)

    The caller provides:
      - ``reward_fn``: callable(xt_hat, context) → float (or list[float] for batched eval)
      - ``diffusion_model``: any object exposing ``.denoise(xt, t, context) -> xt_minus1``
        and ``.predict_x0(xt, t, context) -> x0_hat``

    This design keeps FK steering completely decoupled from the generative backbone
    so it can wrap RFdiffusion, FrameDiff, Chroma, or any other diffusion model.
    """

    def __init__(
        self,
        config: FKSteeringConfig,
        reward_fn: Callable,
        diffusion_model,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.reward_fn = reward_fn
        self.diffusion_model = diffusion_model
        self.device = device
        self.potential_fn = build_potential(config.potential)

        if hasattr(self.potential_fn, "reset"):
            self.potential_fn.reset()

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        context,
        n_timesteps: int = 50,
        init_noise: Optional[torch.Tensor] = None,
    ) -> List[Particle]:
        """
        Run FK-steered reverse diffusion.

        Parameters
        ----------
        context:
            Target structure / hotspot encoding passed through to both the
            diffusion model and the reward function.
        n_timesteps:
            Total number of denoising steps T.
        init_noise:
            Optional pre-sampled Gaussian noise of shape (n_particles, ...).
            If None, the diffusion model's ``sample_prior`` method is called.

        Returns
        -------
        List of :class:`Particle` objects at t=0, sorted by descending reward.
        """
        cfg = self.config
        N = cfg.n_particles

        # ---- initialise particles from prior ----
        if init_noise is not None:
            assert init_noise.shape[0] == N
            noise = init_noise.to(self.device)
        else:
            noise = self.diffusion_model.sample_prior(N, context).to(self.device)

        particles = [Particle(xt=noise[i]) for i in range(N)]
        prev_rewards: Optional[torch.Tensor] = None

        logger.info(
            "FK steering: %d particles, τ=%.1f, Δt=%d, t_start=%d, potential=%s",
            N, cfg.temperature, cfg.resample_interval, cfg.guidance_start, cfg.potential,
        )

        # ---- reverse diffusion loop ----
        for t in range(n_timesteps - 1, -1, -1):
            # 1. Propagate each particle one denoising step
            particles = self._propagate(particles, t, context)

            # 2. FK guidance: evaluate rewards and resample
            if t <= cfg.guidance_start and (t % cfg.resample_interval == 0):
                particles, prev_rewards = self._guidance_step(
                    particles, t, context, prev_rewards
                )

        particles.sort(key=lambda p: p.reward if p.reward is not None else -1e9, reverse=True)
        return particles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _propagate(self, particles: List[Particle], t: int, context) -> List[Particle]:
        """Apply one reverse denoising step to every particle."""
        for p in particles:
            p.xt = self.diffusion_model.denoise(p.xt.unsqueeze(0), t, context).squeeze(0)
        return particles

    def _guidance_step(
        self,
        particles: List[Particle],
        t: int,
        context,
        prev_rewards: Optional[torch.Tensor],
    ) -> Tuple[List[Particle], torch.Tensor]:
        """Evaluate rewards, compute potentials, resample."""
        cfg = self.config

        # ---- reward evaluation ----
        raw_rewards = []
        for p in particles:
            r = self._evaluate_reward(p.xt, t, context)
            p.update_reward(r)
            raw_rewards.append(r)

        rewards = torch.tensor(raw_rewards, dtype=torch.float32)

        # Apply guidance temperature: r_t = r / τ
        scaled_rewards = rewards / cfg.temperature

        # ---- potential ----
        scaled_prev = prev_rewards / cfg.temperature if prev_rewards is not None else None
        potentials = self.potential_fn(scaled_rewards, scaled_prev)

        # ---- numerical stabilisation (clip before exp already done in potential) ----
        potentials = potentials - potentials.max()   # log-space shift
        potentials = torch.exp(potentials)
        weights = potentials / potentials.sum()

        logger.debug(
            "t=%3d  mean_r=%.3f  max_r=%.3f  ess=%.1f",
            t,
            rewards.mean().item(),
            rewards.max().item(),
            self._ess(weights),
        )

        # ---- systematic resampling ----
        indices = self._systematic_resample(weights, cfg.n_particles)
        particles = [deepcopy(particles[i]) for i in indices]

        return particles, scaled_rewards

    def _evaluate_reward(self, xt: torch.Tensor, t: int, context) -> float:
        """
        Compute averaged reward over n_reward_samples ProteinMPNN/relaxation evaluations.

        The diffusion model predicts x̂0|t from the noisy xt, then the reward_fn
        (which internally calls ProteinMPNN + PyRosetta) is invoked n times and
        the results are aggregated.
        """
        cfg = self.config

        with torch.no_grad():
            x0_hat = self.diffusion_model.predict_x0(xt.unsqueeze(0), t, context).squeeze(0)

        sample_rewards = []
        for _ in range(cfg.n_reward_samples):
            try:
                r = self.reward_fn(x0_hat, context)
            except Exception as exc:
                logger.warning("Reward evaluation failed: %s", exc)
                r = 0.0
            sample_rewards.append(float(r))

        if cfg.reward_aggregation == "max":
            return max(sample_rewards)
        return float(np.mean(sample_rewards))

    @staticmethod
    def _systematic_resample(weights: torch.Tensor, n: int) -> List[int]:
        """
        Low-variance systematic resampling (standard SMC algorithm).

        Generates a single uniform offset u ~ Uniform[0, 1/n) then selects
        indices at positions u, u+1/n, u+2/n, ... on the CDF.
        """
        w = weights.cpu().numpy()
        w = w / w.sum()
        cdf = np.cumsum(w)

        u = np.random.uniform(0, 1.0 / n)
        positions = u + np.arange(n) / n

        indices = []
        cumsum_idx = 0
        for pos in positions:
            while cumsum_idx < len(cdf) - 1 and cdf[cumsum_idx] < pos:
                cumsum_idx += 1
            indices.append(cumsum_idx)
        return indices

    @staticmethod
    def _ess(weights: torch.Tensor) -> float:
        """Effective sample size: 1 / sum(w^2), normalised by N."""
        w = weights / weights.sum()
        return float(1.0 / (w ** 2).sum().item())
