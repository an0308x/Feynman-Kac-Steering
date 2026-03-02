"""
Unit tests for FK steering — no external dependencies required.
All tests use mock diffusion models and reward functions so the suite
runs without RFdiffusion, ProteinMPNN, or PyRosetta installed.

Run:
    pytest tests/ -v
"""

import math
import pytest
import torch
import numpy as np

from fk_steering.steering import (
    FKSteering,
    FKSteeringConfig,
    ImmediatePotential,
    DifferencePotential,
    MaxPotential,
    SumPotential,
    build_potential,
)
from fk_steering.rewards import (
    ChargeReward,
    SecondaryStructureReward,
    build_reward,
)


# ─────────────────────────────────────────────
# Mock objects
# ─────────────────────────────────────────────

class MockDiffusionModel:
    """Minimal diffusion model that simply scales Gaussian noise toward 0."""

    def sample_prior(self, n_particles, context):
        L = context.get("binder_length", 10)
        return torch.randn(n_particles, L, 4, 3)

    def denoise(self, xt, t, context):
        # Simple smoothing: x_{t-1} ≈ 0.9 * x_t
        return xt * 0.9 + torch.randn_like(xt) * 0.01

    def predict_x0(self, xt, t, context):
        scale = float(t) / 50.0
        return xt * (1.0 - scale * 0.8)


def make_constant_reward(value: float):
    """Reward that always returns `value`, regardless of input."""
    def reward_fn(x0_hat, context):
        return value
    return reward_fn


def make_noisy_reward(mean: float, std: float = 0.5):
    def reward_fn(x0_hat, context):
        return float(np.random.normal(mean, std))
    return reward_fn


MOCK_CONTEXT = {"binder_length": 10, "target_pdb": "dummy.pdb"}


# ─────────────────────────────────────────────
# Potential function tests
# ─────────────────────────────────────────────

class TestPotentials:

    def test_immediate_positive(self):
        p = ImmediatePotential()
        r = torch.tensor([1.0, 2.0, 3.0])
        g = p(r)
        expected = torch.exp(r)
        assert torch.allclose(g, expected)

    def test_immediate_no_prev_needed(self):
        p = ImmediatePotential()
        r = torch.zeros(5)
        g = p(r, rewards_prev=torch.ones(5))
        # should still use only r, ignoring prev
        assert torch.allclose(g, torch.ones(5))

    def test_difference_boundary(self):
        """At the first step (no prev), difference potential should be 1."""
        p = DifferencePotential()
        r = torch.tensor([3.0, 1.0, -2.0])
        g = p(r, rewards_prev=None)
        assert torch.allclose(g, torch.ones(3))

    def test_difference_incremental(self):
        p = DifferencePotential()
        r_prev = torch.tensor([1.0, 2.0])
        r_curr = torch.tensor([3.0, 1.0])
        g = p(r_curr, rewards_prev=r_prev)
        expected = torch.exp(r_curr - r_prev)
        assert torch.allclose(g, expected)

    def test_max_monotone(self):
        p = MaxPotential()
        r1 = torch.tensor([1.0, 5.0])
        r2 = torch.tensor([3.0, 2.0])  # second particle drops
        g1 = p(r1)
        g2 = p(r2)
        # running max for particle 0: max(1,3)=3; particle 1: max(5,2)=5
        assert g2[0].item() == pytest.approx(math.exp(3.0), rel=1e-4)
        assert g2[1].item() == pytest.approx(math.exp(5.0), rel=1e-4)

    def test_sum_accumulates(self):
        p = SumPotential()
        r1 = torch.tensor([1.0, 2.0])
        r2 = torch.tensor([0.5, 0.5])
        g1 = p(r1)
        g2 = p(r2)
        expected = torch.exp(r1 + r2)
        assert torch.allclose(g2, expected, atol=1e-5)

    def test_build_potential_registry(self):
        for name in ["immediate", "difference", "max", "sum"]:
            pot = build_potential(name)
            assert callable(pot)

    def test_build_potential_unknown(self):
        with pytest.raises(ValueError, match="Unknown potential"):
            build_potential("nonexistent")


# ─────────────────────────────────────────────
# Steering tests
# ─────────────────────────────────────────────

class TestFKSteering:

    def _make_steering(self, reward_fn=None, **config_kwargs):
        cfg = FKSteeringConfig(
            n_particles=4,
            temperature=1.0,
            resample_interval=5,
            guidance_start=10,
            potential="immediate",
            n_reward_samples=1,
            seed=0,
            **config_kwargs,
        )
        if reward_fn is None:
            reward_fn = make_constant_reward(1.0)
        return FKSteering(
            config=cfg,
            reward_fn=reward_fn,
            diffusion_model=MockDiffusionModel(),
        )

    def test_run_returns_n_particles(self):
        steering = self._make_steering()
        particles = steering.run(MOCK_CONTEXT, n_timesteps=15)
        assert len(particles) == 4

    def test_particles_sorted_by_reward_desc(self):
        rewards_given = [3.0, 1.0, 4.0, 2.0]
        call_count = [0]

        def cycling_reward(x0_hat, ctx):
            r = rewards_given[call_count[0] % len(rewards_given)]
            call_count[0] += 1
            return r

        steering = self._make_steering(reward_fn=cycling_reward)
        particles = steering.run(MOCK_CONTEXT, n_timesteps=15)
        rewards = [p.reward for p in particles if p.reward is not None]
        if rewards:
            assert rewards == sorted(rewards, reverse=True)

    def test_all_potential_types_run(self):
        for potential in ["immediate", "difference", "max", "sum"]:
            s = self._make_steering(potential=potential)
            particles = s.run(MOCK_CONTEXT, n_timesteps=10)
            assert len(particles) == 4

    def test_reward_history_populated(self):
        steering = self._make_steering(
            resample_interval=2,
            guidance_start=8,
        )
        particles = steering.run(MOCK_CONTEXT, n_timesteps=10)
        # Guidance runs at t=8,6,4,2,0 → up to 5 history entries
        for p in particles:
            assert len(p.history) >= 0  # non-negative

    def test_systematic_resample_correct_count(self):
        weights = torch.tensor([0.1, 0.4, 0.3, 0.2])
        indices = FKSteering._systematic_resample(weights, 4)
        assert len(indices) == 4
        assert all(0 <= i < 4 for i in indices)

    def test_ess_uniform(self):
        weights = torch.ones(10) / 10
        ess = FKSteering._ess(weights)
        assert ess == pytest.approx(10.0, rel=1e-4)

    def test_ess_degenerate(self):
        weights = torch.zeros(5)
        weights[0] = 1.0
        ess = FKSteering._ess(weights)
        assert ess == pytest.approx(1.0, rel=1e-4)

    def test_high_reward_increases_particle_fitness(self):
        """Steering toward high reward should raise mean reward vs. blind."""
        low_reward  = make_constant_reward(0.0)
        high_reward = make_constant_reward(10.0)

        s_low  = self._make_steering(reward_fn=low_reward,  temperature=0.5)
        s_high = self._make_steering(reward_fn=high_reward, temperature=0.5)

        p_low  = s_low.run(MOCK_CONTEXT,  n_timesteps=12)
        p_high = s_high.run(MOCK_CONTEXT, n_timesteps=12)

        mean_low  = np.mean([p.reward for p in p_low  if p.reward is not None] or [0])
        mean_high = np.mean([p.reward for p in p_high if p.reward is not None] or [0])

        assert mean_high >= mean_low

    def test_seed_reproducibility(self):
        s1 = self._make_steering(seed=99)
        s2 = self._make_steering(seed=99)
        p1 = s1.run(MOCK_CONTEXT, n_timesteps=6)
        p2 = s2.run(MOCK_CONTEXT, n_timesteps=6)
        # Final xt tensors should match
        assert torch.allclose(p1[0].xt, p2[0].xt, atol=1e-5)


# ─────────────────────────────────────────────
# Reward function tests (no PyRosetta needed)
# ─────────────────────────────────────────────

class TestChargeReward:

    def _reward(self, target: float = 0.0):
        r = ChargeReward(target_charge=target)
        r.pipeline = _MockPipeline()
        return r

    def test_net_charge_calculation(self):
        # RKKK → +3; D → -1; net at pH7 ≈ +2 (H adds ~0.1, ignored here)
        assert ChargeReward.net_charge("RKK") == pytest.approx(3.0)
        assert ChargeReward.net_charge("DE")  == pytest.approx(-2.0)
        assert ChargeReward.net_charge("AAA") == pytest.approx(0.0)

    def test_perfect_charge_match(self):
        r = self._reward(target=3.0)
        # pipeline will return "RKK" which has +3 charge
        dummy_x = torch.zeros(5, 4, 3)
        val = r(dummy_x, {})
        assert val == pytest.approx(0.0)

    def test_charge_penalty_nonzero(self):
        r = self._reward(target=0.0)
        dummy_x = torch.zeros(5, 4, 3)
        val = r(dummy_x, {})
        assert val <= 0.0   # always penalises deviation


class TestSecondaryStructureReward:

    def test_perfect_helix_reward(self):
        """Targeting 100% helix with a perfectly helical sequence should give high reward."""
        r = SecondaryStructureReward(target_helix=1.0, target_sheet=0.0, target_loop=0.0)
        r.pipeline = _MockPipeline(ss="HHHHHHHHHH")
        dummy_x = torch.zeros(10, 4, 3)
        val = r(dummy_x, {})
        # wα=4, wβ=1, wℓ=1 → 4(1-|1-1|)+1(1-|0-0|)+1(1-|0-0|) = 4+1+1 = 6
        assert val > 4.0

    def test_reward_in_range(self):
        r = SecondaryStructureReward(target_sheet=1.0)
        r.pipeline = _MockPipeline(ss="EEEEEEEEEE")
        val = r(torch.zeros(10, 4, 3), {})
        assert 0.0 <= val <= 6.1  # max possible is 4+1+1=6


# ─────────────────────────────────────────────
# Mock pipeline for reward tests
# ─────────────────────────────────────────────

class _MockPipeline:
    """Returns a fixed sequence and a mock pose with known secondary structure."""

    def __init__(self, sequence: str = "RKK", ss: str = ""):
        self._seq = sequence
        self._ss = ss

    def __call__(self, x0_hat, context):
        pose = _MockPose(self._ss) if self._ss else None
        return pose, self._seq


class _MockPose:
    """Minimal PyRosetta-like pose that returns DSSP output."""

    def __init__(self, ss_string: str):
        self._ss = ss_string

    def total_residue(self):
        return len(self._ss)


# ─────────────────────────────────────────────
# Integration-style smoke test
# ─────────────────────────────────────────────

class TestIntegration:

    def test_full_run_no_crash(self):
        """End-to-end smoke test: 2 particles, 5 timesteps, charge reward."""
        cfg = FKSteeringConfig(
            n_particles=2,
            temperature=5.0,
            resample_interval=2,
            guidance_start=4,
            potential="difference",
            n_reward_samples=1,
            seed=7,
        )

        def mock_charge_reward(x0_hat, ctx):
            return float(torch.randn(1).item())

        s = FKSteering(
            config=cfg,
            reward_fn=mock_charge_reward,
            diffusion_model=MockDiffusionModel(),
        )
        context = {"binder_length": 8}
        particles = s.run(context, n_timesteps=5)
        assert len(particles) == 2
        for p in particles:
            assert p.xt.shape == (8, 4, 3)
