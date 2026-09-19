# How to run

## Requirements

- **Python 3.10–3.12** (Taichi does not support 3.13+). The commands below pin 3.12
  via `uv`, so no manual environment setup is needed.
- No API keys. No network access at run time. No private paths.

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv sync --extra dev
```

## 1. Watch it (autonomous demonstration)

A judge can run this and understand the argument without typing anything else.

```bash
uv run --python 3.12 --extra dev python -m invisible_planet.app.main
```

Five stages run automatically: the system, transit measurement, the anomaly, the
inference, and the reveal. The O−C panel is computed live from the running
integration.

| Key | Action |
|---|---|
| `SPACE` | pause / resume |
| `N` | skip to the next stage |
| `LMB` drag | orbit the camera |
| `W` / `S` | zoom in / out |
| `ESC` | quit |

`--speed 2` takes more physics steps per frame. It does **not** change the
timestep, which is fixed by the convergence study.

## 2. Investigate it (interactive mode)

```bash
uv run --python 3.12 --extra dev python -m invisible_planet.app.main --investigate
```

Set your own hypothesis for the hidden planet's mass, semi-major axis and phase,
press `R`, and watch whether it reproduces the observed transit timings. The χ²
shown is the same statistic the automated inference minimises. Every control is a
real model parameter.

## 3. Verify it (full scientific validation)

```bash
uv run --python 3.12 --extra dev python scripts/run_validation.py
```

Runs every test, maps results onto the project's validation checklist, and writes
`VALIDATION_RESULTS.md` + `data/validation_report.json`. Exit code 0 means every
requirement passed.

## 4. Reproduce the experiment

```bash
# rebuild the reference dataset from the archive snapshot
uv run --python 3.12 --extra dev python scripts/build_reference_dataset.py

# baseline stability of the real system, with no synthetic body
uv run --python 3.12 --extra dev python scripts/run_baseline_stability.py

# the Planet X parameter sweep and its selection criteria
uv run --python 3.12 --extra dev python scripts/sweep_planet_x.py

# generate observations and run the blind recovery  (~25 min)
uv run --python 3.12 --extra dev python scripts/run_experiment.py
uv run --python 3.12 --extra dev python scripts/run_experiment.py --quick   # faster

# how well is Planet X actually determined?
uv run --python 3.12 --extra dev python scripts/run_identifiability.py
```

## 5. Run the tests directly

```bash
uv run --python 3.12 --extra dev python -m pytest tests/ -q
```

## Performance notes

Physics runs on the **CPU backend in f64**. This is deliberate: Taichi's Vulkan
backend does not support f64, and f32 would fabricate spurious transit timing
variations (see `docs/ARCHITECTURE.md` §3.1). With ~10 bodies this costs nothing —
the parallelism that matters is across *candidate systems* during the parameter
search, which the CPU backend threads well.

A Taichi warning about `f32 <- f64` precision during rendering is expected and
harmless: the graphics vertex buffer is f32 while the physics is f64.
