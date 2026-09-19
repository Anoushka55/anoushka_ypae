"""
Phase 5 gate — the real Kepler-90 initial state and its baseline stability.

These tests check that the initial conditions FAITHFULLY REPRESENT THE DATASET
(rather than merely being self-consistent), and that the system integrates
stably with no synthetic body present.
"""

from __future__ import annotations

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.physics.diagnostics import osculating_elements
from invisible_planet.physics.engine import NBodyEngine
from invisible_planet.systems.kepler90 import (
    PLANET_ORDER,
    build_kepler90,
    load_reference,
    recommended_timestep,
)


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_reference()


@pytest.fixture(scope="module")
def system(reference):
    return build_kepler90(reference=reference)


# =============================================================================
# The initial state must reproduce the dataset
# =============================================================================

def test_system_contains_only_real_bodies(system):
    """The baseline system must contain no synthetic body.

    Phase 5 exists to validate the REAL system. If Planet X leaked in here, every
    later control-vs-mystery comparison would be meaningless.
    """
    assert system.metadata["contains_synthetic_body"] is False
    assert system.n_bodies == 9  # 1 star + 8 known planets
    assert system.names[0] == "Kepler-90"
    for letter in PLANET_ORDER:
        assert f"Kepler-90 {letter}" in system.names


def test_osculating_periods_match_the_catalogue(system, reference):
    """Each planet's initial osculating period must match its catalogue period.

    Tolerance 1e-4 relative. The residual is dominated by a known systematic:
    the catalogue's semi-major axes were derived with a slightly different value
    of GM_sun than the IAU nominal value adopted here, a 1.26e-5 relative offset
    documented in docs/KEPLER90_DATA.md.
    """
    elements = osculating_elements(system.masses, system.positions, system.velocities)
    for letter, el in zip(PLANET_ORDER, elements):
        catalogue = reference["planets"][letter]["orbital_period_days"]["value"]
        relative = abs(el["period"] - catalogue) / catalogue
        assert relative < 1e-4, (
            f"planet {letter}: osculating P = {el['period']:.6f} d vs catalogue "
            f"{catalogue:.6f} d (relative {relative:.2e})"
        )


def test_osculating_axes_follow_keplers_law_with_total_mass(system, reference):
    """Semi-major axis must satisfy Kepler's third law using the TOTAL mass.

    We derive a from the measured period rather than reading the catalogue's a,
    because the catalogue computed a from the STELLAR MASS ALONE. Tolerance
    1e-9 relative against the correct two-body relation.
    """
    elements = osculating_elements(system.masses, system.positions, system.velocities)
    m_star = reference["star"]["mass_solar"]["value"]
    for idx, (letter, el) in enumerate(zip(PLANET_ORDER, elements), start=1):
        period = reference["planets"][letter]["orbital_period_days"]["value"]
        expected = C.kepler_third_law_axis(period, m_star + system.masses[idx])
        relative = abs(el["semi_major_axis"] - expected) / expected
        assert relative < 1e-9, f"planet {letter}: a mismatch, relative {relative:.2e}"


def test_axis_deviation_from_catalogue_is_the_neglected_planet_mass(system, reference):
    """The deviation from the catalogue's a must be explained, not merely tolerated.

    The catalogue used a^3 = G M_star P^2 / 4pi^2, omitting the planet mass. The
    correct relation uses G(M_star + m_p), so our a is larger by a factor
    (1 + m_p/M_star)^(1/3) - 1 ~ m_p/(3 M_star).

    This test asserts that the ENTIRE discrepancy is accounted for by that
    factor, which turns an unexplained mismatch into a quantitative confirmation
    that both we and the catalogue are doing what we claim. For Kepler-90 h the
    effect is 1.7e-4 in a (2.2 hours per orbit in P); for the small planets it is
    ~1e-6 and the residual is dominated by the catalogue's slightly different
    value of GM_sun (a uniform 1.26e-5, see docs/KEPLER90_DATA.md).
    """
    m_star = reference["star"]["mass_solar"]["value"]
    for idx, letter in enumerate(PLANET_ORDER, start=1):
        catalogue = reference["planets"][letter]["semi_major_axis_au"]["value"]
        period = reference["planets"][letter]["orbital_period_days"]["value"]
        ours = C.kepler_third_law_axis(period, m_star + system.masses[idx])

        observed_shift = (ours - catalogue) / catalogue
        predicted_shift = (1.0 + system.masses[idx] / m_star) ** (1.0 / 3.0) - 1.0
        # the residual after removing the planet-mass effect is the known GM_sun
        # offset between our constants and the catalogue's
        residual = abs(observed_shift - predicted_shift)
        assert residual < 3e-5, (
            f"planet {letter}: shift {observed_shift:.3e} not explained by the "
            f"neglected planet mass {predicted_shift:.3e} (residual {residual:.3e})"
        )


@pytest.mark.parametrize("letter", PLANET_ORDER)
def test_planet_is_transiting_at_its_published_transit_epoch(letter, reference):
    """The orbital phase must be inherited correctly from the measured transit epoch.

    Building the system AT a planet's published transit epoch must place that
    planet in front of the star: z > 0 (towards the observer) and a sky-projected
    star-planet separation below the stellar radius.

    This validates the single most delicate conversion in the initial conditions.
    A sign error or a wrong anomaly convention would show up here immediately and
    would otherwise silently corrupt every transit time in the project.
    """
    transit_epoch = reference["planets"][letter]["transit_epoch_bjd"]["value"]
    system = build_kepler90(epoch_bjd=transit_epoch, planets=[letter], reference=reference)

    star_pos = system.positions[0]
    planet_pos = system.positions[1]
    offset = planet_pos - star_pos
    sky_separation = np.hypot(offset[0], offset[1])
    stellar_radius = system.radii[0]

    assert offset[2] > 0.0, f"planet {letter} is behind the star at its transit epoch"
    assert sky_separation < stellar_radius, (
        f"planet {letter} sky separation {sky_separation:.6f} au exceeds the stellar "
        f"radius {stellar_radius:.6f} au at its published transit epoch"
    )


def test_system_is_barycentric(system):
    """Total linear momentum must be zero at t = 0, to round-off."""
    momentum = (system.masses[:, None] * system.velocities).sum(axis=0)
    scale = float(np.max(np.abs(system.masses[:, None] * system.velocities)))
    assert np.max(np.abs(momentum)) / scale < 1e-14


def test_planet_masses_carry_their_provenance(reference):
    """Six of eight planets have model-derived masses; that must remain visible.

    This is a scientific-integrity test, not a numerical one. If a future change
    silently promoted derived masses to 'observed', the project would be
    misrepresenting its inputs.
    """
    statuses = {
        letter: reference["planets"][letter]["mass_earth"]["status"]
        for letter in PLANET_ORDER
    }
    assert statuses["g"] == "observed"
    assert statuses["h"] == "observed"
    for letter in ["b", "c", "i", "d", "e", "f"]:
        assert statuses[letter] == "derived", (
            f"planet {letter} mass is marked '{statuses[letter]}' but no published "
            "mass exists for it"
        )


def test_eccentricity_mode_refuses_negative_archive_values(reference):
    """A negative 'eccentricity' in the archive must be refused, not used.

    Liang et al. (2021) has e = -0.011 for planet h in the archive's scalar
    eccentricity column; that is an eccentricity VECTOR COMPONENT, and using it
    as an eccentricity would be physically meaningless.
    """
    with pytest.raises(ValueError, match="negative eccentricity"):
        build_kepler90(use_measured_eccentricities=True, reference=reference)


# =============================================================================
# Baseline stability
# =============================================================================

@pytest.fixture(scope="module")
def baseline_run(system, reference):
    """Integrate the real system for 400 days and record element evolution.

    400 days keeps the test suite fast; the full 1500-day Kepler baseline is
    covered by scripts/run_baseline_stability.py, whose results are recorded in
    docs/VALIDATION.md.
    """
    dt = recommended_timestep(reference)
    engine = NBodyEngine(n_bodies=system.n_bodies)
    engine.set_state(system.masses, system.positions, system.velocities)

    e0 = float(engine.total_energy()[0])
    n_samples = 80
    chunk = int(400.0 / dt) // n_samples

    axes, eccs, energy_err, separations = [], [], [], []
    for _ in range(n_samples):
        engine.run(dt, chunk)
        d = engine.diagnostics()
        el = osculating_elements(
            system.masses, engine.get_positions()[0], engine.get_velocities()[0]
        )
        axes.append([e["semi_major_axis"] for e in el])
        eccs.append([e["eccentricity"] for e in el])
        energy_err.append(abs(float(d["total_energy"][0]) - e0) / abs(e0))
        separations.append(float(d["min_separation"][0]))

    return {
        "axes": np.array(axes),
        "eccentricities": np.array(eccs),
        "energy_error": np.array(energy_err),
        "min_separation": np.array(separations),
    }


def test_no_orbits_cross(baseline_run):
    """Adjacent orbits must not cross — the physical signature of instability.

    Crossing means the inner planet's apoapsis reaches the outer planet's
    periapsis, which makes a future close encounter possible. This is a far more
    meaningful criterion than a threshold on the SIZE of osculating variations:
    a compact system legitimately shows ~1% oscillations, which is perturbation
    theory working, not instability.
    """
    axes = baseline_run["axes"]
    eccs = baseline_run["eccentricities"]
    for i in range(len(PLANET_ORDER) - 1):
        apoapsis_inner = np.max(axes[:, i] * (1.0 + eccs[:, i]))
        periapsis_outer = np.min(axes[:, i + 1] * (1.0 - eccs[:, i + 1]))
        assert periapsis_outer > apoapsis_inner, (
            f"orbits of {PLANET_ORDER[i]} and {PLANET_ORDER[i+1]} cross: "
            f"Q_in = {apoapsis_inner:.6f} au, q_out = {periapsis_outer:.6f} au"
        )


def test_semi_major_axes_are_bounded_not_drifting(baseline_run):
    """Osculating axes must oscillate, not drift.

    Metric: the linear trend across the run, compared with the oscillation range
    it sits within. A ratio below 1 means the motion is dominated by bounded
    oscillation rather than secular drift.
    """
    axes = baseline_run["axes"]
    for idx, letter in enumerate(PLANET_ORDER):
        series = axes[:, idx]
        span = series.max() - series.min()
        trend = abs(np.polyfit(np.arange(len(series)), series, 1)[0] * len(series))
        assert trend < span, (
            f"planet {letter}: semi-major axis trend {trend:.3e} au exceeds its "
            f"oscillation range {span:.3e} au — possible secular drift"
        )


def test_all_orbits_remain_bound_and_elliptical(baseline_run):
    """No planet may be ejected or driven onto a crossing-eccentricity orbit."""
    assert np.all(baseline_run["axes"] > 0.0), "a planet became unbound"
    assert np.max(baseline_run["eccentricities"]) < 0.1, (
        f"eccentricity reached {np.max(baseline_run['eccentricities']):.3f}, far above "
        "anything supported by the data"
    )


def test_planet_g_perturbation_is_the_largest(baseline_run):
    """Planet g must be the most strongly perturbed body — a physical prediction.

    g is adjacent to h, by far the most massive planet (203 M_earth), and the
    g-h pair is the most tightly spaced in the system at 5.15 mutual Hill radii.
    Perturbation theory therefore predicts g shows the largest osculating
    variation, and that is exactly what the real system shows: the TTVs of
    Kepler-90 g and h are what allowed their masses to be measured at all.

    This test asserts a PHYSICAL EXPECTATION about which body moves most, so a
    future change that scrambled masses or ordering would be caught.
    """
    axes = baseline_run["axes"]
    fractional_range = [
        (axes[:, i].max() - axes[:, i].min()) / axes[0, i] for i in range(len(PLANET_ORDER))
    ]
    most_perturbed = PLANET_ORDER[int(np.argmax(fractional_range))]
    assert most_perturbed == "g", (
        f"expected planet g to be the most perturbed, got {most_perturbed}: "
        f"{dict(zip(PLANET_ORDER, [f'{v:.2e}' for v in fractional_range]))}"
    )


def test_baseline_energy_is_conserved(baseline_run):
    """Relative energy error over the baseline run must stay below 1e-6."""
    assert float(np.max(baseline_run["energy_error"])) < 1e-6


def test_no_close_approach_in_baseline(baseline_run):
    """Minimum pairwise separation must stay well clear of a collision.

    The closest approach in this system is between planets b and c, whose
    orbital radii differ by only 0.0116 au. That is ordinary geometry, not an
    encounter, but it is checked so that a genuine close approach could never
    pass unnoticed with no softening in the force law.
    """
    closest = float(np.min(baseline_run["min_separation"]))
    assert closest > 0.005, f"closest approach {closest:.5f} au"
