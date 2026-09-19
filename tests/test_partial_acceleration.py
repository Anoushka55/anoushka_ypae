"""
The perturbation arrows shown by the UI are one term of the Newtonian force sum. These tests
check that claim against direct arithmetic, so the display cannot imply a force that the
integration does not contain.
"""

from __future__ import annotations

import numpy as np

from invisible_planet import constants as C
from invisible_planet.physics.engine import NBodyEngine


def _random_system(seed: int = 3, n: int = 5):
    rng = np.random.default_rng(seed)
    masses = np.concatenate([[1.0], rng.uniform(1e-6, 1e-4, n - 1)])
    positions = rng.uniform(-1.0, 1.0, (n, 3))
    velocities = rng.normal(0.0, 0.01, (n, 3))
    return masses, positions, velocities


def test_partial_accelerations_sum_to_the_full_acceleration():
    masses, positions, velocities = _random_system()
    engine = NBodyEngine(n_bodies=len(masses))
    engine.set_state(masses, positions, velocities, time=0.0)
    total = sum(engine.acceleration_due_to(j)[0] for j in range(len(masses)))
    engine._compute_accelerations()
    full = engine.acc.to_numpy()[0]
    scale = np.abs(full).max()
    assert np.abs(total - full).max() <= 1e-12 * scale


def test_partial_acceleration_matches_newton_by_hand():
    masses, positions, velocities = _random_system(seed=8)
    engine = NBodyEngine(n_bodies=len(masses))
    engine.set_state(masses, positions, velocities, time=0.0)
    source = 3
    got = engine.acceleration_due_to(source)[0]
    for i in range(len(masses)):
        if i == source:
            assert np.all(got[i] == 0.0)        # a body exerts no force on itself
            continue
        d = positions[source] - positions[i]
        expected = C.G * masses[source] * d / np.linalg.norm(d) ** 3
        assert np.allclose(got[i], expected, rtol=1e-12, atol=0.0)


def test_reading_a_partial_acceleration_does_not_change_the_state():
    masses, positions, velocities = _random_system(seed=5)
    engine = NBodyEngine(n_bodies=len(masses))
    engine.set_state(masses, positions, velocities, time=0.0)
    before = (engine.get_positions().copy(), engine.get_velocities().copy())
    engine.acceleration_due_to(2)
    assert np.array_equal(before[0], engine.get_positions())
    assert np.array_equal(before[1], engine.get_velocities())
