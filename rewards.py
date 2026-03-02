"""
Reward functions for FK-steered protein design
===============================================
All rewards follow the interface:

    reward_fn(x0_hat: torch.Tensor, context: Any) -> float

Each reward function is designed to be passed into :class:`FKSteering` and is
evaluated on the *refined* sequence–structure pair (x̃0|t, st) produced by the
ProteinMPNN + PyRosetta pipeline described in the paper.

Available rewards
-----------------
- :class:`BindingEnergyReward`   — interface ΔG (binder design)
- :class:`ChargeReward`          — net charge at pH 7
- :class:`SecondaryStructureReward` — secondary structure composition
- :func:`build_reward`           — factory function by name

Note on optional dependencies
------------------------------
PyRosetta and ProteinMPNN are *required* for production use but are imported
lazily so the rest of the codebase remains importable for testing/mocking.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseReward(ABC):
    """All rewards must implement __call__."""

    @abstractmethod
    def __call__(self, x0_hat: torch.Tensor, context) -> float:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Sequence recovery pipeline  (ProteinMPNN + PyRosetta)
# ---------------------------------------------------------------------------

class SequenceStructurePipeline:
    """
    Wraps ProteinMPNN sequence design followed by PyRosetta side-chain packing
    and local energy minimisation to produce a physically coherent sequence–
    structure pair (x̃0, s) from a coarse backbone x̂0.

    This corresponds to the ``Pref(· | x0)`` distribution described in the paper
    (Methods: Sequence recovery and packing).

    Parameters
    ----------
    mpnn_temperature:
        Sampling temperature for ProteinMPNN (default 0.2, as in the paper).
    use_solubility_weights:
        Use ProteinMPNN's solubility-optimised model weights.
    n_relax_rounds:
        Number of PyRosetta FastRelax rounds.
    """

    def __init__(
        self,
        mpnn_temperature: float = 0.2,
        use_solubility_weights: bool = True,
        n_relax_rounds: int = 1,
    ) -> None:
        self.mpnn_temperature = mpnn_temperature
        self.use_solubility_weights = use_solubility_weights
        self.n_relax_rounds = n_relax_rounds
        self._mpnn_model = None
        self._rosetta_init = False

    # -- lazy initialisation -------------------------------------------------

    def _ensure_mpnn(self):
        if self._mpnn_model is not None:
            return
        try:
            from protein_mpnn_utils import ProteinMPNN  # ProteinMPNN repo
            self._mpnn_model = ProteinMPNN(
                ca_only=False,
                use_soluble_model=self.use_solubility_weights,
            )
            logger.info("ProteinMPNN loaded.")
        except ImportError:
            logger.warning(
                "ProteinMPNN not found. Install from https://github.com/dauparas/ProteinMPNN. "
                "Reward evaluation will return 0."
            )

    def _ensure_rosetta(self):
        if self._rosetta_init:
            return
        try:
            import pyrosetta
            pyrosetta.init("-mute all")
            self._rosetta_init = True
            logger.info("PyRosetta initialised.")
        except ImportError:
            logger.warning(
                "PyRosetta not found. Install from https://www.pyrosetta.org. "
                "Side-chain packing/relaxation will be skipped."
            )

    # -- public interface ----------------------------------------------------

    def __call__(self, x0_hat: torch.Tensor, context) -> Tuple[Optional[object], Optional[str]]:
        """
        Run the full pipeline:  backbone → sequence (MPNN) → packed/relaxed pose.

        Returns
        -------
        (pose, sequence)  where pose is a PyRosetta Pose and sequence is a str.
        Returns (None, None) if dependencies are unavailable.
        """
        self._ensure_mpnn()
        self._ensure_rosetta()

        coords = x0_hat.detach().cpu().numpy()  # (L, 4, 3) Ca/N/C/O frames

        # --- sequence design ---
        sequence = self._run_mpnn(coords, context)
        if sequence is None:
            return None, None

        # --- build all-atom pose ---
        pose = self._thread_and_relax(coords, sequence)
        return pose, sequence

    def _run_mpnn(self, coords: np.ndarray, context) -> Optional[str]:
        if self._mpnn_model is None:
            return None
        try:
            result = self._mpnn_model.sample(
                coords,
                temperature=self.mpnn_temperature,
                context=context,
            )
            return result["sequence"]
        except Exception as exc:
            logger.warning("ProteinMPNN sampling failed: %s", exc)
            return None

    def _thread_and_relax(self, coords: np.ndarray, sequence: str) -> Optional[object]:
        if not self._rosetta_init:
            return None
        try:
            import pyrosetta
            from pyrosetta.rosetta.protocols.minimization_packing import MinMover
            from pyrosetta.rosetta.core.pack.task import TaskFactory
            from pyrosetta.rosetta.protocols.relax import FastRelax

            pose = pyrosetta.pose_from_sequence(sequence)
            # Thread backbone coordinates
            _thread_coords_into_pose(coords, pose)

            # Pack side-chains
            tf = TaskFactory()
            packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(tf)
            packer.apply(pose)

            # FastRelax
            fr = FastRelax()
            fr.set_scorefxn(pyrosetta.get_fa_scorefxn())
            fr.max_iter(self.n_relax_rounds * 5)
            fr.apply(pose)

            return pose
        except Exception as exc:
            logger.warning("PyRosetta relax failed: %s", exc)
            return None


def _thread_coords_into_pose(coords: np.ndarray, pose) -> None:
    """Copy Cα positions from an (L, 3) or (L, 4, 3) array into a PyRosetta pose."""
    try:
        import pyrosetta
        from pyrosetta.rosetta.numeric import xyzVector_double_t as Vec

        n_res = coords.shape[0]
        ca_coords = coords[:, 1] if coords.ndim == 3 else coords  # prefer Cα channel
        for i in range(min(n_res, pose.total_residue())):
            xyz = Vec(float(ca_coords[i, 0]), float(ca_coords[i, 1]), float(ca_coords[i, 2]))
            pose.residue(i + 1).set_xyz("CA", xyz)
    except Exception as exc:
        logger.debug("Coordinate threading skipped: %s", exc)


# ---------------------------------------------------------------------------
# Binding energy reward  (rbind = -ΔG)
# ---------------------------------------------------------------------------

class BindingEnergyReward(BaseReward):
    """
    Interface free energy reward for binder design.

    r_bind = -ΔG_interface

    where ΔG is computed via PyRosetta's InterfaceAnalyzer on the
    binder–target complex after sequence design and relaxation.

    Parameters
    ----------
    pipeline:
        Shared SequenceStructurePipeline instance.
    target_chain:
        Chain ID of the fixed receptor (default "A").
    binder_chain:
        Chain ID of the designed binder (default "B").
    """

    def __init__(
        self,
        pipeline: Optional[SequenceStructurePipeline] = None,
        target_chain: str = "A",
        binder_chain: str = "B",
    ) -> None:
        self.pipeline = pipeline or SequenceStructurePipeline()
        self.target_chain = target_chain
        self.binder_chain = binder_chain

    def __call__(self, x0_hat: torch.Tensor, context) -> float:
        pose, sequence = self.pipeline(x0_hat, context)
        if pose is None:
            logger.debug("Pipeline unavailable — returning 0 for binding reward.")
            return 0.0
        return self._compute_ddg(pose)

    def _compute_ddg(self, pose) -> float:
        try:
            import pyrosetta
            from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

            interface = f"{self.binder_chain}_{self.target_chain}"
            analyzer = InterfaceAnalyzerMover(interface)
            analyzer.set_scorefunction(pyrosetta.get_fa_scorefxn())
            analyzer.apply(pose)
            dG = analyzer.get_interface_dG()
            return -float(dG)   # paper maximises -ΔG
        except Exception as exc:
            logger.warning("InterfaceAnalyzer failed: %s", exc)
            return 0.0


# ---------------------------------------------------------------------------
# Charge reward  (rcharge = -|Q - Q*|)
# ---------------------------------------------------------------------------

# Residue net charges at pH 7 (canonical values)
_RESIDUE_CHARGE_PH7: Dict[str, float] = {
    "R": +1.0, "K": +1.0,   # positive
    "D": -1.0, "E": -1.0,   # negative
    "H": +0.1,               # ~10% protonated at pH 7
    # all others ≈ 0
}


class ChargeReward(BaseReward):
    """
    Penalises deviation from a target net charge Q* at pH 7.

    r_charge = -|Q - Q*|

    Parameters
    ----------
    target_charge:
        Desired net charge Q*.
    pipeline:
        SequenceStructurePipeline; only the sequence output is used.
    """

    def __init__(
        self,
        target_charge: float = 0.0,
        pipeline: Optional[SequenceStructurePipeline] = None,
    ) -> None:
        self.target_charge = target_charge
        self.pipeline = pipeline or SequenceStructurePipeline()

    def __call__(self, x0_hat: torch.Tensor, context) -> float:
        _, sequence = self.pipeline(x0_hat, context)
        if sequence is None:
            return 0.0
        q = sum(_RESIDUE_CHARGE_PH7.get(aa, 0.0) for aa in sequence.upper())
        return -abs(q - self.target_charge)

    @staticmethod
    def net_charge(sequence: str) -> float:
        return sum(_RESIDUE_CHARGE_PH7.get(aa, 0.0) for aa in sequence.upper())


# ---------------------------------------------------------------------------
# Secondary structure reward
# ---------------------------------------------------------------------------

# DSSP single-letter codes grouped by class
_HELIX_CODES = {"H", "G", "I"}
_SHEET_CODES = {"E", "B"}
_LOOP_CODES  = {"T", "S", "C", " ", "-"}

# Residue-level propensities from Chou-Fasman (simplified)
_HELIX_PROPENSITY: Dict[str, float] = {
    "A": 1.45, "L": 1.34, "M": 1.30, "E": 1.53, "Q": 1.17,
    "K": 1.16, "R": 1.21, "H": 1.00, "V": 1.14, "I": 1.00,
    "Y": 0.74, "C": 0.77, "W": 1.02, "F": 1.12, "T": 0.82,
    "G": 0.53, "N": 0.73, "P": 0.59, "S": 0.79, "D": 0.98,
}
_SHEET_PROPENSITY: Dict[str, float] = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.37,
    "L": 1.22, "T": 1.19, "C": 1.30, "M": 1.05, "Q": 1.10,
    "A": 0.97, "R": 0.90, "G": 0.81, "D": 0.80, "K": 0.74,
    "S": 0.72, "H": 0.87, "N": 0.65, "P": 0.62, "E": 0.52,
}


class SecondaryStructureReward(BaseReward):
    """
    Rewards designs whose secondary structure composition matches user-specified
    target fractions (α*, β*, loop*).

    r_SS = wα(1 - |α - α*|) + wβ(1 - |β - β*|) + wℓ(1 - |ℓ - ℓ*|)

    As in the paper, the targeted class is weighted 4× relative to the others.

    Parameters
    ----------
    target_helix:
        Target α-helix fraction (0–1).
    target_sheet:
        Target β-sheet fraction (0–1).
    target_loop:
        Target loop fraction (0–1).
    dssp_weight:
        Weight of DSSP signal vs sequence propensity (paper uses 0.8).
    pipeline:
        SequenceStructurePipeline.
    """

    def __init__(
        self,
        target_helix: float = 0.0,
        target_sheet: float = 0.0,
        target_loop: float = 0.0,
        dssp_weight: float = 0.8,
        pipeline: Optional[SequenceStructurePipeline] = None,
    ) -> None:
        self.target_helix = target_helix
        self.target_sheet = target_sheet
        self.target_loop = target_loop
        self.dssp_weight = dssp_weight
        self.pipeline = pipeline or SequenceStructurePipeline()

        # Determine which class is the primary target for 4× weighting
        targets = {"helix": target_helix, "sheet": target_sheet, "loop": target_loop}
        self._primary = max(targets, key=targets.get)

    def __call__(self, x0_hat: torch.Tensor, context) -> float:
        pose, sequence = self.pipeline(x0_hat, context)
        alpha, beta, loop = self._compute_fractions(pose, sequence)
        return self._score(alpha, beta, loop)

    def _compute_fractions(
        self, pose, sequence: Optional[str]
    ) -> Tuple[float, float, float]:
        """Blend DSSP (0.8) and sequence propensity (0.2) estimates."""
        dssp_alpha = dssp_beta = dssp_loop = None

        if pose is not None:
            try:
                from pyrosetta.rosetta.core.scoring.dssp import Dssp
                dssp = Dssp(pose)
                ss = dssp.get_dssp_secstruct()
                n = len(ss)
                dssp_alpha = sum(1 for c in ss if c in _HELIX_CODES) / n
                dssp_beta  = sum(1 for c in ss if c in _SHEET_CODES) / n
                dssp_loop  = sum(1 for c in ss if c in _LOOP_CODES)  / n
            except Exception as exc:
                logger.debug("DSSP failed: %s", exc)

        seq_alpha = seq_beta = seq_loop = 1 / 3
        if sequence:
            n = len(sequence)
            seq_alpha = np.mean([_HELIX_PROPENSITY.get(aa, 1.0) for aa in sequence.upper()])
            seq_beta  = np.mean([_SHEET_PROPENSITY.get(aa, 1.0) for aa in sequence.upper()])
            # Normalise
            total = seq_alpha + seq_beta
            seq_alpha /= total; seq_beta /= total
            seq_loop = 1.0 - seq_alpha - seq_beta

        if dssp_alpha is not None:
            w = self.dssp_weight
            alpha = w * dssp_alpha + (1 - w) * seq_alpha
            beta  = w * dssp_beta  + (1 - w) * seq_beta
            loop  = w * dssp_loop  + (1 - w) * seq_loop
        else:
            alpha, beta, loop = seq_alpha, seq_beta, seq_loop

        return alpha, beta, loop

    def _score(self, alpha: float, beta: float, loop: float) -> float:
        base_w = 1.0
        wα = 4.0 if self._primary == "helix" else base_w
        wβ = 4.0 if self._primary == "sheet" else base_w
        wℓ = 4.0 if self._primary == "loop"  else base_w

        return (
            wα * (1 - abs(alpha - self.target_helix)) +
            wβ * (1 - abs(beta  - self.target_sheet)) +
            wℓ * (1 - abs(loop  - self.target_loop))
        )


# ---------------------------------------------------------------------------
# Composite / combined reward
# ---------------------------------------------------------------------------

class CompositeReward(BaseReward):
    """
    Linearly combines multiple reward functions.

    Parameters
    ----------
    rewards_and_weights:
        List of (reward_fn, weight) tuples.
    """

    def __init__(self, rewards_and_weights) -> None:
        self.components = rewards_and_weights

    def __call__(self, x0_hat: torch.Tensor, context) -> float:
        total = 0.0
        for fn, w in self.components:
            total += w * fn(x0_hat, context)
        return total


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_reward(name: str, **kwargs) -> BaseReward:
    registry = {
        "binding":    BindingEnergyReward,
        "charge":     ChargeReward,
        "secondary":  SecondaryStructureReward,
    }
    name = name.lower()
    if name not in registry:
        raise ValueError(f"Unknown reward '{name}'. Choose from {list(registry)}")
    return registry[name](**kwargs)
