"""
RFdiffusion adapter
===================
Thin wrapper around RFdiffusion that exposes the three-method interface
required by :class:`fk_steering.steering.FKSteering`:

    model.sample_prior(N, context)   -> Tensor (N, ...)
    model.denoise(xt, t, context)    -> Tensor
    model.predict_x0(xt, t, context) -> Tensor

This adapter keeps FK steering completely decoupled from RFdiffusion's internal
API so it can be replaced with any other diffusion backbone (FrameDiff, Chroma,
Genie, …) by writing a new adapter.

RFdiffusion installation
------------------------
Clone https://github.com/RosettaCommons/RFdiffusion and follow its README.
Then point ``RFDIFFUSION_PATH`` (env var or argument) at the repo root.

The adapter is imported lazily — if RFdiffusion is not installed the rest of
the codebase still works (useful for unit-testing with mock models).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class RFdiffusionAdapter:
    """
    Wraps a loaded RFdiffusion model to satisfy the FKSteering model protocol.

    Parameters
    ----------
    rfdiffusion_path:
        Path to the cloned RFdiffusion repository.  Falls back to the
        ``RFDIFFUSION_PATH`` environment variable.
    checkpoint:
        Model checkpoint to load.  Defaults to the beta-sheet checkpoint
        used in the paper's binder-design campaigns.
    device:
        Torch device string.
    """

    BETA_SHEET_CKPT  = "models/BFF_ckpt.pt"
    DEFAULT_CKPT     = "models/Base_ckpt.pt"

    def __init__(
        self,
        rfdiffusion_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self._model = None
        self._conf  = None

        # Resolve repo path
        repo = rfdiffusion_path or os.environ.get("RFDIFFUSION_PATH")
        if repo is None:
            logger.warning(
                "RFDIFFUSION_PATH not set.  RFdiffusion adapter will not load a real model.  "
                "Set the environment variable or pass rfdiffusion_path=... to use real models."
            )
            return

        repo = Path(repo).expanduser()
        if not repo.exists():
            raise FileNotFoundError(f"RFdiffusion path does not exist: {repo}")

        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        ckpt_name = checkpoint or self.BETA_SHEET_CKPT
        ckpt_path = repo / ckpt_name
        if not ckpt_path.exists():
            ckpt_path = repo / self.DEFAULT_CKPT

        self._load(ckpt_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, ckpt_path: Path) -> None:
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
            from rfdiffusion.inference.utils import sampler_and_target_from_config  # noqa
            import rfdiffusion.inference.model_runners as model_runners

            logger.info("Loading RFdiffusion checkpoint: %s", ckpt_path)
            # Minimal OmegaConf config mirroring RFdiffusion defaults
            self._conf = OmegaConf.create({
                "inference": {
                    "ckpt_override_path": str(ckpt_path),
                    "num_designs": 1,
                    "schedule_directory": "config/schedules",
                },
                "denoiser": {"noise_scale_ca": 1.0, "noise_scale_frame": 1.0},
                "diffuser": {
                    "T": 50,
                    "so3": {"min_sigma": 0.02, "max_sigma": 1.5, "schedule_type": "linear"},
                    "r3":  {"min_b": 0.01, "max_b": 7.0, "coordinate_scaling": 0.25},
                },
                "potentials": {"guiding_potentials": [], "guide_scale": 1.0},
                "ppi": {},
                "contigmap": {"contigs": ["100"]},
            })
            self._runner = model_runners.SelfConditioningRunner(self._conf)
            logger.info("RFdiffusion loaded successfully.")
        except Exception as exc:
            logger.warning("RFdiffusion failed to load (%s).  Using mock model.", exc)
            self._runner = None

    # ------------------------------------------------------------------
    # FKSteering model protocol
    # ------------------------------------------------------------------

    def sample_prior(self, n_particles: int, context: Any) -> torch.Tensor:
        """
        Sample the prior distribution p(xT) — pure Gaussian noise in SE(3).

        Returns a tensor of shape (n_particles, L, 4, 3) representing per-residue
        rigid frames (N, Cα, C, O positions) for a binder of length L.
        """
        L = _extract_binder_length(context)
        # Independent Gaussian noise in R3 (RFdiffusion rescales internally)
        noise = torch.randn(n_particles, L, 4, 3)
        return noise

    def denoise(self, xt: torch.Tensor, t: int, context: Any) -> torch.Tensor:
        """
        Run one reverse denoising step:  p_θ(x_{t-1} | x_t, context).

        Parameters
        ----------
        xt:
            Noisy structure tensor of shape (1, L, 4, 3).
        t:
            Current timestep (counting from 0 = clean).
        context:
            Dict with at minimum ``target_pdb`` (path or structure tensor) and
            ``hotspot_residues`` (list of residue indices).
        """
        if self._runner is not None:
            return self._rfdiffusion_step(xt, t, context)
        # Mock: simple Gaussian smoothing toward zero
        scale = float(t) / 50.0
        return xt * (1 - 0.05 * scale) + torch.randn_like(xt) * 0.01

    def predict_x0(self, xt: torch.Tensor, t: int, context: Any) -> torch.Tensor:
        """
        Predict the clean structure x̂0|t = f_θ(x_t, t).

        This is the denoised *proxy* on which rewards are evaluated at
        intermediate diffusion steps (before final sampling).
        """
        if self._runner is not None:
            return self._rfdiffusion_predict_x0(xt, t, context)
        # Mock: just return xt with reduced noise
        scale = float(t) / 50.0
        return xt * (1.0 - scale * 0.8)

    # ------------------------------------------------------------------
    # Internal RFdiffusion calls
    # ------------------------------------------------------------------

    def _rfdiffusion_step(self, xt: torch.Tensor, t: int, context: Any) -> torch.Tensor:
        """Thin wrapper calling RFdiffusion's denoiser for one step."""
        try:
            # RFdiffusion runners accept batch dicts; we construct the minimal one
            batch = _build_rfdiffusion_batch(xt, t, context, device=self.device)
            with torch.no_grad():
                out = self._runner.model(batch)
            return out["final_frames"].cpu()
        except Exception as exc:
            logger.warning("RFdiffusion denoising step failed: %s.  Returning xt.", exc)
            return xt

    def _rfdiffusion_predict_x0(self, xt: torch.Tensor, t: int, context: Any) -> torch.Tensor:
        """Use RFdiffusion's single-step prediction of the clean structure."""
        try:
            batch = _build_rfdiffusion_batch(xt, t, context, device=self.device)
            with torch.no_grad():
                out = self._runner.model(batch)
            return out.get("pred_x0", out["final_frames"]).cpu()
        except Exception as exc:
            logger.warning("RFdiffusion x0 prediction failed: %s.  Returning xt.", exc)
            return xt


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_binder_length(context) -> int:
    if isinstance(context, dict):
        return int(context.get("binder_length", 24))
    if hasattr(context, "binder_length"):
        return int(context.binder_length)
    return 24


def _build_rfdiffusion_batch(xt: torch.Tensor, t: int, context, device: str = "cpu") -> dict:
    """
    Construct a minimal RFdiffusion-compatible batch dict from an FK particle.

    In a full integration this would unpack target chain coordinates, hotspot
    residue masks, etc. from the context object.  Here we provide the skeleton
    that a complete RFdiffusion integration would fill in.
    """
    L_binder = xt.shape[1]
    L_target = 0
    target_coords = None

    if isinstance(context, dict):
        target_coords = context.get("target_coords")  # (L_target, 4, 3) tensor or None
        if target_coords is not None:
            L_target = target_coords.shape[0]

    L_total = L_binder + L_target
    t_tensor = torch.tensor([t], dtype=torch.long, device=device)

    batch = {
        "t": t_tensor,
        "noisy_frames": xt.to(device),
        "seq_len": torch.tensor([L_total], device=device),
    }

    if target_coords is not None:
        batch["target_frames"] = target_coords.unsqueeze(0).to(device)

    if isinstance(context, dict) and "hotspot_residues" in context:
        hotspots = context["hotspot_residues"]
        mask = torch.zeros(L_total, dtype=torch.bool, device=device)
        for idx in hotspots:
            if idx < L_total:
                mask[idx] = True
        batch["hotspot_mask"] = mask.unsqueeze(0)

    return batch
