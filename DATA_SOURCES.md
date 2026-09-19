# Data Sources

Generated from `data/sources_registry.json` (machine-readable registry). Archive snapshot accessed 2026-09-19T06:03:26Z. Planet X is synthetic and appears in no source.

## Status vocabulary

- **observed**: Measured from data and published with an uncertainty.
- **inferred**: Obtained from data through a model fit (e.g. dynamical TTV mass).
- **adopted-model**: Assumed value, not measured (e.g. e = 0).
- **derived**: Computed from other quantities by a cited relation.
- **synthetic**: Invented for this numerical experiment (Planet X only).
- **numerical**: A property of the integration, not of nature (e.g. timestep).

## Adoption rules

| Parameter | Source | Status | Rationale |
|---|---|---|---|
| orbital_period_days | WEISS_ET_AL__2024 | observed | Only source covering all 8 planets with a single self-consistent solution. |
| semi_major_axis_au | WEISS_ET_AL__2024 | derived | Published by Weiss et al.; verified here to follow Kepler's third law with their own stellar mass, so it is a derived quantity, not an independent measurement. |
| planet_radius_earth | WEISS_ET_AL__2024 | observed | Transit depth x stellar radius; same source as the orbital solution. |
| eccentricity | WEISS_ET_AL__2024 | adopted-model | Weiss et al. fix e = 0 for all planets (no uncertainties quoted). Treated as an ADOPTED MODELLING ASSUMPTION, not a measurement. Measured values exist for g and h only and are recorded as alternates. |
| inclination_deg | CABRERA_ET_AL__2014 | observed | Cabrera et al. for b-h; Shallue & Vanderburg 2018 for i. CROSS-SOURCE: flagged. |
| transit_epoch_bjd | CABRERA_ET_AL__2014 | observed | Cabrera et al. for b-h; Shallue & Vanderburg 2018 for i. Sets orbital phase. CROSS-SOURCE: flagged. |

## Bibliography

- **WEISS_ET_AL__2024**: Weiss et al. (2024), ApJS 270, 8 — The Kepler Giant Planet Search (https://ui.adsabs.harvard.edu/abs/2024ApJS..270....8W/abstract)
- **CABRERA_ET_AL__2014**: Cabrera et al. (2014), ApJ 781, 18 — The planetary system to KIC 11442793: a compact analogue to the Solar System (https://ui.adsabs.harvard.edu/abs/2014ApJ...781...18C/abstract)
- **SHALLUE__AMP__VANDERBURG_2018**: Shallue & Vanderburg (2018), AJ 155, 94 — Identifying Exoplanets with Deep Learning: discovery of Kepler-90 i (https://ui.adsabs.harvard.edu/abs/2018AJ....155...94S/abstract)
- **LIANG_ET_AL__2021**: Liang et al. (2021), AJ 161, 202 — transit-timing analysis of Kepler-90 (https://ui.adsabs.harvard.edu/abs/2021AJ....161..202L/abstract)
- **SHAW_ET_AL__2025**: Shaw et al. (2025), AJ 170, 146 — transit-timing analysis of Kepler-90 (https://ui.adsabs.harvard.edu/abs/2025AJ....170..146S/abstract)
- **FULTON__AMP__PETIGURA_2018**: Fulton & Petigura (2018), AJ 156, 264 — The California-Kepler Survey VII (homogeneous stellar parameters) (https://ui.adsabs.harvard.edu/abs/2018AJ....156..264F/abstract)
- **WEISS__AMP__MARCY_2014**: Weiss & Marcy (2014), ApJ 783, L6 — The Mass-Radius Relation for 65 Exoplanets Smaller than 4 Earth Radii (https://ui.adsabs.harvard.edu/abs/2014ApJ...783L...6W/abstract)
- **NASA_EXOPLANET_ARCHIVE**: NASA Exoplanet Archive, Planetary Systems (ps) table, TAP service (https://exoplanetarchive.ipac.caltech.edu/)

## Per-parameter registry

| Parameter | Value | Unit | Uncertainty | Status | Source |
|---|---|---|---|---|---|
| star.mass_solar | 1.108 | M_sun | 0.035 | observed | FULTON__AMP__PETIGURA_2018 |
| star.radius_solar | 1.185 | R_sun | 0.031 | observed | FULTON__AMP__PETIGURA_2018 |
| star.teff_k | 6015.0 | K | None | observed | FULTON__AMP__PETIGURA_2018 |
| planet.b.orbital_period_days | 7.00828076 | days | None | observed | WEISS_ET_AL__2024 |
| planet.b.semi_major_axis_au | 0.0741641627 | au | None | derived | WEISS_ET_AL__2024 |
| planet.b.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.b.inclination_deg | 89.4 | deg | 1.5 | observed | CABRERA_ET_AL__2014 |
| planet.b.impact_parameter | 0.13 | stellar radii | None | observed | CABRERA_ET_AL__2014 |
| planet.b.transit_epoch_bjd | 2454970.6906 | BJD_TDB | 0.0017 | observed | CABRERA_ET_AL__2014 |
| planet.b.planet_radius_earth | 1.3 | R_earth | 0.065 | observed | WEISS_ET_AL__2024 |
| planet.b.mass_earth | 2.7335449007321295 | M_earth | 1.3667724503660648 | derived | WEISS__AMP__MARCY_2014 |
| planet.c.orbital_period_days | 8.71982853 | days | None | observed | WEISS_ET_AL__2024 |
| planet.c.semi_major_axis_au | 0.0857942931 | au | None | derived | WEISS_ET_AL__2024 |
| planet.c.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.c.inclination_deg | 89.68 | deg | 0.74 | observed | CABRERA_ET_AL__2014 |
| planet.c.impact_parameter | 0.09 | stellar radii | None | observed | CABRERA_ET_AL__2014 |
| planet.c.transit_epoch_bjd | 2454972.5687 | BJD_TDB | 0.0023 | observed | CABRERA_ET_AL__2014 |
| planet.c.planet_radius_earth | 1.445 | R_earth | 0.066 | observed | WEISS_ET_AL__2024 |
| planet.c.mass_earth | 4.0239462558780215 | M_earth | 2.0119731279390107 | derived | WEISS__AMP__MARCY_2014 |
| planet.i.orbital_period_days | 14.44912 | days | None | observed | WEISS_ET_AL__2024 |
| planet.i.semi_major_axis_au | 0.1201380843 | au | None | derived | WEISS_ET_AL__2024 |
| planet.i.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.i.inclination_deg | 89.2 | deg | 0.59 | observed | SHALLUE__AMP__VANDERBURG_2018 |
| planet.i.impact_parameter | 0.5 | stellar radii | None | observed | SHALLUE__AMP__VANDERBURG_2018 |
| planet.i.transit_epoch_bjd | 2455644.3488 | BJD_TDB | 0.0048 | observed | SHALLUE__AMP__VANDERBURG_2018 |
| planet.i.planet_radius_earth | 1.32 | R_earth | 0.21 | observed | WEISS_ET_AL__2024 |
| planet.i.mass_earth | 2.890037462079402 | M_earth | 1.445018731039701 | derived | WEISS__AMP__MARCY_2014 |
| planet.d.orbital_period_days | 59.7370348 | days | None | observed | WEISS_ET_AL__2024 |
| planet.d.semi_major_axis_au | 0.309467855 | au | None | derived | WEISS_ET_AL__2024 |
| planet.d.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.d.inclination_deg | 89.71 | deg | 0.29 | observed | CABRERA_ET_AL__2014 |
| planet.d.impact_parameter | 0.28 | stellar radii | None | observed | CABRERA_ET_AL__2014 |
| planet.d.transit_epoch_bjd | 2454991.9656 | BJD_TDB | 0.0042 | observed | CABRERA_ET_AL__2014 |
| planet.d.planet_radius_earth | 2.866 | R_earth | 0.094 | observed | WEISS_ET_AL__2024 |
| planet.d.mass_earth | 7.161749609290917 | M_earth | 4.3 | derived | WEISS__AMP__MARCY_2014 |
| planet.e.orbital_period_days | 91.9404603 | days | None | observed | WEISS_ET_AL__2024 |
| planet.e.semi_major_axis_au | 0.412531915 | au | None | derived | WEISS_ET_AL__2024 |
| planet.e.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.e.inclination_deg | 89.79 | deg | 0.19 | observed | CABRERA_ET_AL__2014 |
| planet.e.impact_parameter | 0.27 | stellar radii | None | observed | CABRERA_ET_AL__2014 |
| planet.e.transit_epoch_bjd | 2454967.3127 | BJD_TDB | 0.0063 | observed | CABRERA_ET_AL__2014 |
| planet.e.planet_radius_earth | 2.612 | R_earth | 0.094 | observed | WEISS_ET_AL__2024 |
| planet.e.mass_earth | 6.5695758395945285 | M_earth | 4.3 | derived | WEISS__AMP__MARCY_2014 |
| planet.f.orbital_period_days | 124.916337 | days | None | observed | WEISS_ET_AL__2024 |
| planet.f.semi_major_axis_au | 0.5060567991 | au | None | derived | WEISS_ET_AL__2024 |
| planet.f.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.f.inclination_deg | 89.77 | deg | 0.31 | observed | CABRERA_ET_AL__2014 |
| planet.f.impact_parameter | 0.35 | stellar radii | None | observed | CABRERA_ET_AL__2014 |
| planet.f.transit_epoch_bjd | 2455087.704 | BJD_TDB | 0.014 | observed | CABRERA_ET_AL__2014 |
| planet.f.planet_radius_earth | 2.77 | R_earth | 0.12 | observed | WEISS_ET_AL__2024 |
| planet.f.mass_earth | 6.93838613022816 | M_earth | 4.3 | derived | WEISS__AMP__MARCY_2014 |
| planet.g.orbital_period_days | 210.6031105 | days | None | observed | WEISS_ET_AL__2024 |
| planet.g.semi_major_axis_au | 0.7168522035 | au | None | derived | WEISS_ET_AL__2024 |
| planet.g.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.g.inclination_deg | 89.8 | deg | 0.06 | observed | CABRERA_ET_AL__2014 |
| planet.g.impact_parameter | 0.45 | stellar radii | None | observed | CABRERA_ET_AL__2014 |
| planet.g.transit_epoch_bjd | 2454980.0364 | BJD_TDB | 0.0014 | observed | CABRERA_ET_AL__2014 |
| planet.g.planet_radius_earth | 7.718 | R_earth | 0.206 | observed | WEISS_ET_AL__2024 |
| planet.g.mass_earth | 15.0 | M_earth | 0.9 | observed | LIANG_ET_AL__2021 |
| planet.h.orbital_period_days | 331.6011081 | days | None | observed | WEISS_ET_AL__2024 |
| planet.h.semi_major_axis_au | 0.9702055452 | au | None | derived | WEISS_ET_AL__2024 |
| planet.h.eccentricity | 0.0 | dimensionless | None | adopted-model | WEISS_ET_AL__2024 |
| planet.h.inclination_deg | 89.6 | deg | 1.3 | observed | CABRERA_ET_AL__2014 |
| planet.h.impact_parameter | 0.36 | stellar radii | None | observed | CABRERA_ET_AL__2014 |
| planet.h.transit_epoch_bjd | 2454973.49631 | BJD_TDB | 0.00082 | observed | CABRERA_ET_AL__2014 |
| planet.h.planet_radius_earth | 11.252 | R_earth | 0.306 | observed | WEISS_ET_AL__2024 |
| planet.h.mass_earth | 203.0 | M_earth | 5.0 | observed | LIANG_ET_AL__2021 |