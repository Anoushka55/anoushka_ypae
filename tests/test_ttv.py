"""
Phase 6 gate — transit detection and transit timing variations.

This suite settles the question deferred from Phase 3.2: whether the production
timestep delivers transit times accurate enough for the science, measured
end-to-end through the actual analysis pipeline rather than argued by proxy.

It also establishes the project's core scientific claim in miniature: an
isolated planet produces flat O-C residuals, and gravitational perturbation
produces structured ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from invisible_planet import constants as C
from invisible_planet.observations.transits import observe_transits
from invisible_planet.observations.ttv import (
    assign_epochs,
    compare_ephemerides,
    fit_linear_ephemeris,
)
from invisible_planet.systems.kepler90 import (
    PLANET_ORDER,
    build_kepler90,
    load_reference,
    recommended_timestep,
)

BASELINE_DAYS = 600.0


@pytest.fixture(scope="module")
def reference():
    return load_reference()


# =============================================================================
# Ephemeris machinery
# =============================================================================

def test_linear_ephemeris_recovers_a_known_line_exactly():
    """Fitting a perfectly periodic sequence must return its parameters exactly."""
    t0_true, period_true = 123.456, 7.008281
    epochs = np.arange(40)
    times = t0_true + period_true * epochs

    fit = fit_linear_ephemeris(times)
    assert abs(fit.t0 - t0_true) < 1e-10
    assert abs(fit.period - period_true) < 1e-12
    assert fit.rms_seconds < 1e-6


def test_epoch_assignment_survives_gaps():
    """Missing transits must not shift the epoch numbering.

    Real sequences have data gaps; if epochs were assigned as 0,1,2,... by array
    index, a single missing transit would corrupt every subsequent residual.
    """
    period = 14.44912
    epochs_true = np.array([0, 1, 2, 5, 6, 10, 11, 12, 20])
    times = 100.0 + period * epochs_true

    assert np.array_equal(assign_epochs(times, period), epochs_true)
    fit = fit_linear_ephemeris(times)
    assert fit.rms_seconds < 1e-6
    assert abs(fit.period - period) < 1e-12


def test_two_transits_are_refused():
    """Two points define a line exactly, so their residuals carry no information."""
    with pytest.raises(ValueError, match="at least 3"):
        fit_linear_ephemeris(np.array([10.0, 20.0]))


# =============================================================================
# Transit detection accuracy — the deferred Phase 3.2 question
# =============================================================================

@pytest.mark.parametrize("letter", ["b", "i", "e", "h"])
def test_isolated_planet_has_negligible_oc_residuals(letter, reference):
    """An ISOLATED planet must produce flat O-C residuals — well below one second.

    THIS IS THE DECISIVE NUMERICAL TEST OF THE WHOLE PROJECT, and the reason
    Phase 3.2 deliberately left its timestep verdict open.

    A single planet orbiting a single star is exactly Keplerian, so its transits
    are exactly periodic and the O-C residuals are zero by construction. Any
    residual measured here is pure error from (a) the integrator and (b) the
    transit-centre root find. Because the transit times feed directly into the
    inference, this error must be far smaller than the TTV signals we intend to
    detect, which are minutes in size.

    Requirement: RMS residual < 1 s at the production timestep.
    """
    system = build_kepler90(planets=[letter], reference=reference)
    dt = recommended_timestep(reference)
    period = reference["planets"][letter]["orbital_period_days"]["value"]
    duration = max(BASELINE_DAYS, 6.0 * period)

    series = observe_transits(system, duration, dt)
    times = series.times[0]
    assert len(times) >= 3, f"only {len(times)} transits detected for planet {letter}"

    fit = fit_linear_ephemeris(times)
    assert fit.rms_seconds < 1.0, (
        f"isolated planet {letter}: O-C RMS = {fit.rms_seconds:.4f} s "
        f"(peak-to-peak {fit.peak_to_peak_minutes * 60:.4f} s) over "
        f"{len(times)} transits — transit times are not accurate enough"
    )


def test_fitted_period_offset_is_the_known_symplectic_frequency_shift(reference):
    """The timestep-dependent part of the fitted period is the O(dt^2) symplectic frequency shift.

    A symplectic integrator follows a nearby "shadow" Hamiltonian whose orbital frequency differs
    from the true one at O(dt^2) (docs/NUMERICAL_METHODS.md).

    The fitted period ALSO differs from the catalogue period by a dt-independent amount (the
    catalogue semi-major axis omits the planet's mass and uses the 4 pi^2 shortcut for G M, and the
    osculating period differs from the mean transit period; see docs/UNITS.md). That constant is
    not the subject of this test and would mask the scaling if compared with the catalogue.

    So the test isolates the dt-dependent part by DIFFERENCING successive halvings, which removes
    any constant: P(dt) - P(dt/2) = C dt^2 (1 - 1/4) and P(dt/2) - P(dt/4) = C dt^2 (1/4 - 1/16),
    so the ratio of the two differences must be 4 for a second-order method. A coding error (wrong
    mu, wrong epoch, bad root find) would not obey that scaling.

    The magnitude is bounded by the requirement that the shift at the production timestep be
    below 1e-5 of the period (the shift is fitted away as part of the linear ephemeris, exactly as
    with real data, and verified to cancel in the O-C test below).
    """
    letter = "i"
    system = build_kepler90(planets=[letter], reference=reference)
    base_dt = recommended_timestep(reference)

    periods = []
    for divisor in [1.0, 2.0, 4.0]:
        series = observe_transits(system, BASELINE_DAYS, base_dt / divisor)
        periods.append(fit_linear_ephemeris(series.times[0]).period)

    d1, d2 = periods[0] - periods[1], periods[1] - periods[2]
    ratio = d1 / d2
    assert 3.5 < ratio < 4.5, (
        f"period shift scales as {ratio:.2f}x per halving, not 4x - not the expected O(dt^2) "
        f"behaviour. periods: {periods}"
    )
    shift_at_production_dt = abs(d1) * (4.0 / 3.0) / periods[0]      # C dt^2 = (P(dt) - P(dt/2)) * 4/3
    assert shift_at_production_dt < 1e-5, f"symplectic period shift {shift_at_production_dt:.2e} larger than expected"


def test_oc_residuals_are_insensitive_to_timestep(reference):
    """The O-C RESIDUALS — the actual signal — must not depend on the timestep.

    The raw transit times DO shift with dt, growing linearly with epoch, because
    of the constant frequency offset described above (halving dt changes the
    accumulated shift by ~170 s over 200 days for planet b). That drift is
    absorbed by the ephemeris fit.

    What must be reproducible is the residual structure, since that is what the
    inference reads. Requirement: agreement better than 1 s per transit.
    """
    letter = "b"
    system = build_kepler90(planets=[letter], reference=reference)
    dt = recommended_timestep(reference)

    coarse = fit_linear_ephemeris(observe_transits(system, 200.0, dt).times[0])
    fine = fit_linear_ephemeris(observe_transits(system, 200.0, dt / 2.0).times[0])

    n = min(len(coarse.residuals), len(fine.residuals))
    difference = np.abs(coarse.residuals[:n] - fine.residuals[:n]) * C.SECONDS_PER_DAY
    assert difference.max() < 1.0, (
        f"O-C residuals moved by up to {difference.max():.4f} s when the timestep "
        "was halved — the signal itself is timestep-dependent"
    )


def test_multiplanet_oc_residuals_are_reproducible_below_the_measurement_floor(reference):
    """With real perturbations, O-C residuals must be reproducible far below the
    precision of the real measurements.

    An isolated planet has zero residuals, so agreement there is a weak test.
    Repeating it for the full system is harder: mutual perturbations make the
    system weakly chaotic, so two integrations at different timesteps drift
    apart slowly. Measured here as ~1.6 s over 300 days against a physical
    signal of ~106 s RMS.

    THE RIGHT YARDSTICK IS THE DATA, NOT AN ARBITRARY THRESHOLD. The published
    transit-epoch uncertainties for this system range from 71 s (planet h) to
    1210 s (planet f). A numerical reproducibility of ~1.6 s is more than an
    order of magnitude below the most precise measurement that exists, so it
    cannot influence any inference drawn from timing data of that quality.

    Requirement: numerical reproducibility better than 20% of the SMALLEST
    published transit-epoch uncertainty in the dataset.
    """
    system = build_kepler90(reference=reference)
    dt = recommended_timestep(reference)

    coarse = observe_transits(system, 300.0, dt)
    fine = observe_transits(system, 300.0, dt / 2.0)

    name = "Kepler-90 b"
    fit_coarse = fit_linear_ephemeris(coarse.for_planet(name))
    fit_fine = fit_linear_ephemeris(fine.for_planet(name))

    n = min(len(fit_coarse.residuals), len(fit_fine.residuals))
    difference = np.abs(fit_coarse.residuals[:n] - fit_fine.residuals[:n]) * C.SECONDS_PER_DAY
    signal = fit_coarse.rms_seconds

    published_uncertainties = [
        reference["planets"][letter]["transit_epoch_bjd"]["uncertainty"] * C.SECONDS_PER_DAY
        for letter in PLANET_ORDER
        if reference["planets"][letter]["transit_epoch_bjd"]["uncertainty"] is not None
    ]
    best_measurement = min(published_uncertainties)

    assert difference.max() < 0.2 * best_measurement, (
        f"O-C residuals moved by up to {difference.max():.3f} s under timestep "
        f"refinement; signal RMS {signal:.1f} s; best published transit-epoch "
        f"uncertainty {best_measurement:.1f} s"
    )
    # and the numerical error must be small compared with the signal itself
    assert difference.max() < 0.05 * signal, (
        f"numerical reproducibility {difference.max():.3f} s is "
        f"{difference.max() / signal:.1%} of the signal RMS {signal:.1f} s"
    )


def test_detected_transits_are_actually_on_the_stellar_disk(reference):
    """Every recorded transit must have an impact parameter below 1 stellar radius."""
    system = build_kepler90(reference=reference)
    series = observe_transits(system, BASELINE_DAYS, recommended_timestep(reference))
    stellar_radius = system.radii[0]

    for name, impacts in zip(series.names, series.impact_parameters):
        if len(impacts) == 0:
            continue
        assert np.max(impacts) < stellar_radius, (
            f"{name}: recorded a 'transit' at impact parameter "
            f"{np.max(impacts) / stellar_radius:.3f} stellar radii"
        )


def test_transit_counts_match_the_orbital_periods(reference):
    """Number of transits must equal the span divided by the period, within one."""
    system = build_kepler90(reference=reference)
    series = observe_transits(system, BASELINE_DAYS, recommended_timestep(reference))

    for letter, name in zip(PLANET_ORDER, series.names):
        period = reference["planets"][letter]["orbital_period_days"]["value"]
        expected = BASELINE_DAYS / period
        actual = len(series.for_planet(name))
        assert abs(actual - expected) <= 1.5, (
            f"{name}: {actual} transits over {BASELINE_DAYS} d, expected ~{expected:.1f}"
        )


# =============================================================================
# The physical signal
# =============================================================================

@pytest.fixture(scope="module")
def full_system_series(reference):
    system = build_kepler90(reference=reference)
    return observe_transits(system, BASELINE_DAYS, recommended_timestep(reference))


def test_mutual_perturbations_produce_measurable_ttvs(full_system_series):
    """The full 8-planet system must show O-C structure far above the noise floor.

    This is the project's premise demonstrated on the REAL system: mutual
    gravity alone, with no synthetic body, already produces timing residuals
    orders of magnitude larger than the ~1 s numerical floor established above.
    """
    fit = fit_linear_ephemeris(full_system_series.for_planet("Kepler-90 b"))
    assert fit.rms_seconds > 5.0, (
        f"expected measurable TTVs in the full system, got RMS {fit.rms_seconds:.3f} s"
    )


def test_ttv_signal_dwarfs_the_numerical_floor(reference, full_system_series):
    """Physical TTVs must exceed the numerical error by a wide margin.

    Compares, for the same planet, the O-C RMS with all 8 planets present
    against the O-C RMS when that planet is isolated. The ratio is the
    signal-to-numerical-noise of the measurement, and it must be large for any
    inference built on these residuals to be meaningful.
    """
    letter = "b"
    name = f"Kepler-90 {letter}"
    dt = recommended_timestep(reference)

    isolated = observe_transits(
        build_kepler90(planets=[letter], reference=reference), BASELINE_DAYS, dt
    )
    numerical_floor = fit_linear_ephemeris(isolated.times[0]).rms_seconds
    physical = fit_linear_ephemeris(full_system_series.for_planet(name)).rms_seconds

    assert physical > 20.0 * max(numerical_floor, 1e-3), (
        f"physical TTV RMS {physical:.3f} s is not clearly above the numerical "
        f"floor {numerical_floor:.4f} s"
    )


def test_removing_a_perturber_changes_the_residuals(reference):
    """Causality check in miniature, ahead of the full Phase 7 experiment.

    Planet g's TTVs must change when its dominant perturber (h) is removed. If
    they did not, the residuals would not be caused by gravity at all.
    """
    dt = recommended_timestep(reference)
    with_h = observe_transits(
        build_kepler90(planets=["g", "h"], reference=reference), 900.0, dt
    )
    without_h = observe_transits(
        build_kepler90(planets=["g"], reference=reference), 900.0, dt
    )

    rms_with = fit_linear_ephemeris(with_h.for_planet("Kepler-90 g")).rms_seconds
    rms_without = fit_linear_ephemeris(without_h.times[0]).rms_seconds

    assert rms_with > 10.0 * max(rms_without, 1e-3), (
        f"removing planet h barely changed g's O-C: {rms_with:.3f} s with, "
        f"{rms_without:.4f} s without"
    )


def test_compare_ephemerides_detects_no_difference_for_identical_input():
    """Comparing a sequence with itself must yield exactly zero difference."""
    times = 100.0 + 7.0 * np.arange(30) + 0.001 * np.sin(np.arange(30))
    result = compare_ephemerides(times, times)
    assert result["rms_seconds"] == 0.0
    assert result["detrended_rms_seconds"] == 0.0
