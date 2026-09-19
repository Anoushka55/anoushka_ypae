"""
Phase 12 — the complete scientific validation suite, in one command.

Runs every test, maps the results onto the project's required validation
checklist, and writes a machine-readable report plus VALIDATION_RESULTS.md.

Run:
  uv run --python 3.12 --extra dev python scripts/run_validation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUNIT = ROOT / "data" / "junit.xml"

#: The validation checklist required by the project contract, mapped onto the
#: tests that actually establish each item. A requirement with no test behind it
#: is reported as MISSING rather than quietly assumed.
CHECKLIST = {
    "1. Newtonian two-body force": [
        "test_acceleration_matches_newtons_law",
        "test_newtons_third_law",
    ],
    "2. Circular orbit": [
        "test_circular_orbit_maintains_radius",
        "test_circular_speed_matches_analytic_value",
    ],
    "3. Elliptical orbit": [
        "test_trajectory_matches_analytic_kepler_propagation",
    ],
    "4. Kepler's third law": [
        "test_keplers_third_law_from_measured_period",
        "test_osculating_periods_match_the_catalogue",
    ],
    "5. Vis-viva": [
        "test_vis_viva_holds_along_the_orbit",
    ],
    "6. Energy conservation": [
        "test_secular_energy_drift_is_negligible",
        "test_energy_error_stays_bounded_in_nbody",
        "test_energy_error_is_bounded_not_secular",
    ],
    "7. Linear momentum conservation": [
        "test_linear_momentum_and_centre_of_mass_are_conserved",
        "test_linear_momentum_stays_zero_in_nbody",
    ],
    "8. Angular momentum conservation": [
        "test_angular_momentum_is_conserved",
        "test_angular_momentum_is_conserved_in_nbody",
        "test_angular_momentum_is_conserved_to_machine_precision",
    ],
    "9. Timestep convergence": [
        "test_position_error_is_second_order",
        "test_velocity_error_is_second_order",
        "test_energy_oscillation_is_second_order",
        "test_integrator_is_not_first_order",
        "test_convergence_table_is_monotonic",
    ],
    "10. N-body interaction": [
        "test_taichi_diagnostics_match_independent_numpy_implementation",
        "test_ensemble_members_are_independent",
        "test_planet_g_perturbation_is_the_largest",
    ],
    "11. Control vs mystery causality": [
        "test_zero_mass_planet_x_leaves_transit_times_unchanged",
        "test_signal_grows_monotonically_with_planet_x_mass",
        "test_visible_planets_are_identical_relative_to_the_star",
        "test_planet_x_obeys_the_same_force_law_as_every_other_body",
    ],
    "12. Transit detection": [
        "test_planet_is_transiting_at_its_published_transit_epoch",
        "test_detected_transits_are_actually_on_the_stellar_disk",
        "test_transit_counts_match_the_orbital_periods",
    ],
    "13. O-C calculation": [
        "test_linear_ephemeris_recovers_a_known_line_exactly",
        "test_epoch_assignment_survives_gaps",
        "test_isolated_planet_has_negligible_oc_residuals",
        "test_two_transits_are_refused",
    ],
    "14. Planet X recovery": [
        "test_forward_model_reproduces_the_truth_at_the_true_parameters",
        "test_true_parameters_beat_a_clearly_wrong_candidate",
        "test_search_improves_on_the_null_hypothesis",
    ],
    "15. Noise sensitivity": [
        "test_noise_amplitudes_come_from_the_published_uncertainties",
        "test_noise_is_reproducible_from_its_seed",
        "test_noise_has_the_requested_amplitude",
    ],
    "16. Stability constraints": [
        "test_no_orbits_cross",
        "test_semi_major_axes_are_bounded_not_drifting",
        "test_all_orbits_remain_bound_and_elliptical",
        "test_no_unphysical_close_approach_occurred",
        "test_no_close_approach_in_baseline",
    ],
    "17. Ground-truth isolation (project-specific)": [
        "test_observer_dataset_has_no_field_for_the_hidden_body",
        "test_observer_dataset_contains_no_forbidden_identifier",
        "test_observer_dataset_contains_no_parameter_value_of_the_hidden_body",
        "test_leakage_detector_actually_catches_a_leak",
        "test_inference_module_does_not_import_the_experiment_config",
    ],
    "18. Scientific-integrity of the inputs (project-specific)": [
        "test_planet_masses_carry_their_provenance",
        "test_eccentricity_mode_refuses_negative_archive_values",
        "test_planet_x_parameters_are_labelled_synthetic",
        "test_system_contains_only_real_bodies",
    ],
}


def main() -> None:
    JUNIT.parent.mkdir(parents=True, exist_ok=True)
    print("Running the full scientific validation suite...\n")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", f"--junitxml={JUNIT}", "-q", "-p",
         "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(completed.stdout[-2500:])
    if completed.returncode not in (0, 1):
        print(completed.stderr[-2000:])

    tree = ET.parse(JUNIT)
    results = {}
    for case in tree.iter("testcase"):
        name = case.get("name", "")
        base = name.split("[")[0]
        failed = any(child.tag in ("failure", "error") for child in case)
        skipped = any(child.tag == "skipped" for child in case)
        status = "FAIL" if failed else ("SKIP" if skipped else "PASS")
        # a parametrised test passes only if every case passes
        if base in results and results[base] == "FAIL":
            continue
        results[base] = status

    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")

    print(f"\n{'=' * 74}")
    print("VALIDATION CHECKLIST")
    print("=" * 74)
    checklist_report = {}
    all_ok = True
    for requirement, test_names in CHECKLIST.items():
        statuses = []
        for test_name in test_names:
            statuses.append((test_name, results.get(test_name, "MISSING")))
        verdict = "PASS"
        if any(s == "MISSING" for _, s in statuses):
            verdict = "MISSING"
        if any(s == "FAIL" for _, s in statuses):
            verdict = "FAIL"
        all_ok = all_ok and verdict == "PASS"
        checklist_report[requirement] = {
            "verdict": verdict,
            "tests": {name: status for name, status in statuses},
        }
        print(f"  [{verdict:>7}] {requirement}")
        for name, status in statuses:
            if status != "PASS":
                print(f"            {status}: {name}")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pytest_exit_code": completed.returncode,
        "tests_total": total,
        "tests_passed": passed,
        "all_requirements_met": bool(all_ok and completed.returncode == 0),
        "checklist": checklist_report,
        "per_test": results,
    }
    (ROOT / "data" / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Validation Results",
        "",
        f"**Generated:** {report['generated_utc']}",
        f"**Tests:** {passed}/{total} passed",
        f"**All contract requirements met:** "
        f"{'YES' if report['all_requirements_met'] else 'NO'}",
        "",
        "Reproduce with:",
        "",
        "```",
        "uv run --python 3.12 --extra dev python scripts/run_validation.py",
        "```",
        "",
        "## Required checklist",
        "",
        "| # | Requirement | Verdict | Tests |",
        "|---|---|---|---|",
    ]
    for requirement, record in checklist_report.items():
        number, _, title = requirement.partition(". ")
        lines.append(
            f"| {number} | {title} | **{record['verdict']}** | "
            f"{len(record['tests'])} |"
        )
    lines += [
        "",
        "## Every test",
        "",
        "| Test | Status |",
        "|---|---|",
    ]
    for name in sorted(results):
        lines.append(f"| `{name}` | {results[name]} |")

    (ROOT / "VALIDATION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{passed}/{total} tests passed")
    print("WROTE VALIDATION_RESULTS.md and data/validation_report.json")
    sys.exit(0 if report["all_requirements_met"] else 1)


if __name__ == "__main__":
    main()
