"""
fk_binder_design.py
===================
Command-line entry point for FK-steered protein binder design.

Reproduces the binder design campaigns from:
    Hartman et al. (2025) "Controllable protein design through Feynman-Kac steering"
    arXiv:2511.09216v1

Default parameters match Table 2 / Methods §"Campaigns" in the paper:
    - Immediate potential
    - 50 particles
    - Resample every 2 steps
    - Guidance starts at t=20
    - τ = 10
    - Interface ΔG reward

Usage
-----
python scripts/fk_binder_design.py \\
    --target   targets/4HSC.pdb \\
    --hotspots A110 A115 A117 \\
    --binder_length 24 \\
    --n_particles 50 \\
    --output results/SLO/

Full option reference:
    python scripts/fk_binder_design.py --help
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("fk_binder_design")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FK-steered RFdiffusion binder design",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Target ---
    tgt = p.add_argument_group("Target")
    tgt.add_argument("--target",     required=True, help="Path to target PDB file")
    tgt.add_argument("--hotspots",   nargs="+",     help="Hotspot residues, e.g. A110 A115 A117")
    tgt.add_argument("--binder_length", type=int, default=24, help="Length of designed binder")

    # --- FK steering ---
    fk = p.add_argument_group("FK steering")
    fk.add_argument("--n_particles",       type=int,   default=50)
    fk.add_argument("--temperature",       type=float, default=10.0,
                    help="Guidance temperature τ (lower = stronger guidance)")
    fk.add_argument("--resample_interval", type=int,   default=2,  dest="resample_interval",
                    help="Resample every Δt steps")
    fk.add_argument("--guidance_start",    type=int,   default=20, dest="guidance_start",
                    help="Start guidance at timestep t_start")
    fk.add_argument("--potential",         default="immediate",
                    choices=["immediate", "difference", "max", "sum"],
                    help="FK potential function")
    fk.add_argument("--n_reward_samples",  type=int,   default=5,
                    help="MPNN/relax evaluations per reward estimate")
    fk.add_argument("--n_timesteps",       type=int,   default=50,
                    help="Total reverse diffusion steps T")

    # --- Reward ---
    rwd = p.add_argument_group("Reward")
    rwd.add_argument("--reward",           default="binding",
                     choices=["binding", "charge", "secondary"],
                     help="Reward function type")
    rwd.add_argument("--target_charge",    type=float, default=0.0,
                     help="Target net charge (charge reward only)")
    rwd.add_argument("--target_helix",     type=float, default=0.0,
                     help="Target α-helix fraction (secondary reward)")
    rwd.add_argument("--target_sheet",     type=float, default=0.0,
                     help="Target β-sheet fraction (secondary reward)")
    rwd.add_argument("--target_loop",      type=float, default=1.0,
                     help="Target loop fraction (secondary reward)")

    # --- Model ---
    mdl = p.add_argument_group("Model")
    mdl.add_argument("--rfdiffusion_path", default=os.environ.get("RFDIFFUSION_PATH"),
                     help="Path to RFdiffusion repo (or set RFDIFFUSION_PATH env var)")
    mdl.add_argument("--checkpoint",       default=None,
                     help="RFdiffusion checkpoint (default: beta-sheet)")
    mdl.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")

    # --- Output ---
    out = p.add_argument_group("Output")
    out.add_argument("--output",   default="results/", help="Output directory")
    out.add_argument("--n_output", type=int,   default=10,
                     help="Number of top designs to save")
    out.add_argument("--seed",     type=int,   default=42)
    out.add_argument("--verbose",  action="store_true")

    return p


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(args: argparse.Namespace) -> dict:
    """Load target PDB and build the context dict for the FK steering loop."""
    ctx = {
        "target_pdb":     args.target,
        "binder_length":  args.binder_length,
        "hotspot_residues": [],
    }

    # Parse hotspot residues
    if args.hotspots:
        hotspot_indices = []
        for h in args.hotspots:
            # e.g. "A110" → chain A, residue 110
            chain = h[0] if h[0].isalpha() else "A"
            resnum = int(h.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            hotspot_indices.append({"chain": chain, "resnum": resnum})
        ctx["hotspot_residues"] = hotspot_indices
        logger.info("Hotspots: %s", hotspot_indices)

    # Load target coordinates if PyRosetta is available
    try:
        import pyrosetta
        pyrosetta.init("-mute all")
        pose = pyrosetta.pose_from_pdb(args.target)
        import torch, numpy as np
        # Extract Cα coordinates  (L_target, 3)
        n_res = pose.total_residue()
        ca_coords = np.zeros((n_res, 4, 3))
        for i in range(1, n_res + 1):
            r = pose.residue(i)
            for j, atom in enumerate(["N", "CA", "C", "O"]):
                try:
                    xyz = r.xyz(atom)
                    ca_coords[i-1, j] = [xyz.x, xyz.y, xyz.z]
                except Exception:
                    pass
        ctx["target_coords"] = torch.tensor(ca_coords, dtype=torch.float32)
        logger.info("Loaded target: %d residues from %s", n_res, args.target)
    except Exception as exc:
        logger.warning("Could not load target PDB with PyRosetta (%s). Running without coords.", exc)

    return ctx


# ---------------------------------------------------------------------------
# Design campaign
# ---------------------------------------------------------------------------

def run_design(args: argparse.Namespace) -> None:
    from fk_steering.steering import FKSteering, FKSteeringConfig
    from fk_steering.rewards import (
        BindingEnergyReward, ChargeReward, SecondaryStructureReward,
        SequenceStructurePipeline,
    )
    from fk_steering.rfdiffusion_adapter import RFdiffusionAdapter

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----
    logger.info("Loading RFdiffusion model…")
    model = RFdiffusionAdapter(
        rfdiffusion_path=args.rfdiffusion_path,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    # ---- Reward ----
    pipeline = SequenceStructurePipeline()
    if args.reward == "binding":
        reward_fn = BindingEnergyReward(pipeline=pipeline)
    elif args.reward == "charge":
        reward_fn = ChargeReward(target_charge=args.target_charge, pipeline=pipeline)
    else:
        reward_fn = SecondaryStructureReward(
            target_helix=args.target_helix,
            target_sheet=args.target_sheet,
            target_loop=args.target_loop,
            pipeline=pipeline,
        )
    logger.info("Reward function: %s", reward_fn)

    # ---- FK steering config ----
    config = FKSteeringConfig(
        n_particles      = args.n_particles,
        temperature      = args.temperature,
        resample_interval= args.resample_interval,
        guidance_start   = args.guidance_start,
        potential        = args.potential,
        n_reward_samples = args.n_reward_samples,
        seed             = args.seed,
    )

    steering = FKSteering(config=config, reward_fn=reward_fn, diffusion_model=model,
                          device=args.device)

    # ---- Context ----
    context = build_context(args)

    # ---- Run ----
    logger.info(
        "Starting FK-steered design: %d particles, potential=%s, t_start=%d",
        args.n_particles, args.potential, args.guidance_start,
    )
    t0 = time.perf_counter()
    particles = steering.run(context, n_timesteps=args.n_timesteps)
    elapsed = time.perf_counter() - t0
    logger.info("Design complete in %.1f s", elapsed)

    # ---- Save top designs ----
    top = particles[: args.n_output]
    results = []
    for rank, p in enumerate(top):
        entry = {
            "rank": rank + 1,
            "reward": p.reward,
            "reward_history": p.history,
        }
        # Save structure tensor
        torch.save(p.xt, output_dir / f"design_{rank+1:03d}.pt")
        results.append(entry)
        logger.info("  Rank %2d  reward=%.3f", rank + 1, p.reward or 0)

    # Save JSON summary
    summary = {
        "target":    args.target,
        "hotspots":  args.hotspots,
        "config":    vars(config),
        "n_designs": len(particles),
        "elapsed_s": elapsed,
        "top_designs": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s/summary.json", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_design(args)


if __name__ == "__main__":
    main()
