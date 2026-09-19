# Validation Results

**Generated:** 2026-09-19T07:50:11+00:00
**Tests:** 85/85 passed
**All contract requirements met:** YES

Reproduce with:

```
uv run --python 3.12 --extra dev python scripts/run_validation.py
```

## Required checklist

| # | Requirement | Verdict | Tests |
|---|---|---|---|
| 1 | Newtonian two-body force | **PASS** | 2 |
| 2 | Circular orbit | **PASS** | 2 |
| 3 | Elliptical orbit | **PASS** | 1 |
| 4 | Kepler's third law | **PASS** | 2 |
| 5 | Vis-viva | **PASS** | 1 |
| 6 | Energy conservation | **PASS** | 3 |
| 7 | Linear momentum conservation | **PASS** | 2 |
| 8 | Angular momentum conservation | **PASS** | 3 |
| 9 | Timestep convergence | **PASS** | 5 |
| 10 | N-body interaction | **PASS** | 3 |
| 11 | Control vs mystery causality | **PASS** | 4 |
| 12 | Transit detection | **PASS** | 3 |
| 13 | O-C calculation | **PASS** | 4 |
| 14 | Planet X recovery | **PASS** | 3 |
| 15 | Noise sensitivity | **PASS** | 3 |
| 16 | Stability constraints | **PASS** | 5 |
| 17 | Ground-truth isolation (project-specific) | **PASS** | 5 |
| 18 | Scientific-integrity of the inputs (project-specific) | **PASS** | 4 |

## Every test

| Test | Status |
|---|---|
| `test_acceleration_matches_newtons_law` | PASS |
| `test_all_orbits_remain_bound_and_elliptical` | PASS |
| `test_angular_momentum_is_conserved` | PASS |
| `test_angular_momentum_is_conserved_in_nbody` | PASS |
| `test_angular_momentum_is_conserved_to_machine_precision` | PASS |
| `test_axis_deviation_from_catalogue_is_the_neglected_planet_mass` | PASS |
| `test_barycentric_difference_is_a_pure_galilean_shift` | PASS |
| `test_baseline_energy_is_conserved` | PASS |
| `test_both_systems_are_barycentric` | PASS |
| `test_centre_of_mass_does_not_drift` | PASS |
| `test_chi_squared_is_zero_for_a_perfect_match` | PASS |
| `test_chi_squared_matches_on_epoch_not_index` | PASS |
| `test_circular_orbit_maintains_radius` | PASS |
| `test_circular_speed_matches_analytic_value` | PASS |
| `test_compare_ephemerides_detects_no_difference_for_identical_input` | PASS |
| `test_conservation_report` | PASS |
| `test_control_system_has_no_synthetic_body` | PASS |
| `test_convergence_table_is_monotonic` | PASS |
| `test_dataset_survives_a_save_load_round_trip` | PASS |
| `test_detected_transits_are_actually_on_the_stellar_disk` | PASS |
| `test_eccentricity_mode_refuses_negative_archive_values` | PASS |
| `test_elements_state_roundtrip` | PASS |
| `test_energy_error_does_not_grow_secularly_in_nbody` | PASS |
| `test_energy_error_is_bounded_not_secular` | PASS |
| `test_energy_error_stays_bounded_in_nbody` | PASS |
| `test_energy_oscillation_is_consistent_with_second_order_truncation` | PASS |
| `test_energy_oscillation_is_second_order` | PASS |
| `test_ensemble_members_are_independent` | PASS |
| `test_epoch_assignment_survives_gaps` | PASS |
| `test_fitted_period_offset_is_the_known_symplectic_frequency_shift` | PASS |
| `test_forward_model_reproduces_the_truth_at_the_true_parameters` | PASS |
| `test_four_pi_squared_shortcut_is_rejected` | PASS |
| `test_gravitational_constant_matches_gaussian_constant` | PASS |
| `test_inference_module_does_not_import_the_experiment_config` | PASS |
| `test_integrator_is_not_first_order` | PASS |
| `test_isolated_planet_has_negligible_oc_residuals` | PASS |
| `test_kepler_equation_solution_satisfies_its_own_equation` | PASS |
| `test_keplers_third_law_from_measured_period` | PASS |
| `test_leakage_detector_actually_catches_a_leak` | PASS |
| `test_linear_ephemeris_recovers_a_known_line_exactly` | PASS |
| `test_linear_momentum_and_centre_of_mass_are_conserved` | PASS |
| `test_linear_momentum_stays_zero_in_nbody` | PASS |
| `test_multiplanet_oc_residuals_are_reproducible_below_the_measurement_floor` | PASS |
| `test_mutual_perturbations_produce_measurable_ttvs` | PASS |
| `test_newtons_third_law` | PASS |
| `test_no_close_approach_in_baseline` | PASS |
| `test_no_orbits_cross` | PASS |
| `test_no_unphysical_close_approach_occurred` | PASS |
| `test_noise_amplitudes_come_from_the_published_uncertainties` | PASS |
| `test_noise_has_the_requested_amplitude` | PASS |
| `test_noise_is_reproducible_from_its_seed` | PASS |
| `test_observer_dataset_contains_no_forbidden_identifier` | PASS |
| `test_observer_dataset_contains_no_parameter_value_of_the_hidden_body` | PASS |
| `test_observer_dataset_has_no_field_for_the_hidden_body` | PASS |
| `test_oc_residuals_are_insensitive_to_timestep` | PASS |
| `test_osculating_axes_follow_keplers_law_with_total_mass` | PASS |
| `test_osculating_periods_match_the_catalogue` | PASS |
| `test_phase_error_grows_linearly_in_time` | PASS |
| `test_planet_g_perturbation_is_the_largest` | PASS |
| `test_planet_is_transiting_at_its_published_transit_epoch` | PASS |
| `test_planet_masses_carry_their_provenance` | PASS |
| `test_planet_x_obeys_the_same_force_law_as_every_other_body` | PASS |
| `test_planet_x_parameters_are_labelled_synthetic` | PASS |
| `test_planet_x_produces_a_measurable_timing_signature` | PASS |
| `test_position_error_is_second_order` | PASS |
| `test_removing_a_perturber_changes_the_residuals` | PASS |
| `test_search_improves_on_the_null_hypothesis` | PASS |
| `test_search_is_deterministic_given_its_seed` | PASS |
| `test_secular_energy_drift_is_far_below_the_oscillation` | PASS |
| `test_secular_energy_drift_is_negligible` | PASS |
| `test_semi_major_axes_are_bounded_not_drifting` | PASS |
| `test_signal_grows_monotonically_with_planet_x_mass` | PASS |
| `test_system_contains_only_real_bodies` | PASS |
| `test_system_is_barycentric` | PASS |
| `test_taichi_diagnostics_match_independent_numpy_implementation` | PASS |
| `test_trajectory_matches_analytic_kepler_propagation` | PASS |
| `test_transit_counts_match_the_orbital_periods` | PASS |
| `test_transit_geometry_convention` | PASS |
| `test_true_parameters_beat_a_clearly_wrong_candidate` | PASS |
| `test_ttv_signal_dwarfs_the_numerical_floor` | PASS |
| `test_two_transits_are_refused` | PASS |
| `test_velocity_error_is_second_order` | PASS |
| `test_vis_viva_holds_along_the_orbit` | PASS |
| `test_visible_planets_are_identical_relative_to_the_star` | PASS |
| `test_zero_mass_planet_x_leaves_transit_times_unchanged` | PASS |