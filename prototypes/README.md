# Prototypes

Unrelated to The Invisible Planet. Kept because it is prior work, and moved out of
the repository root so it cannot be mistaken for part of the experiment.

- `galaxy_collision_prototype.py` — a real-time Milky Way / Andromeda collision
  visualiser (restricted N-body: a macro-particle gravity skeleton plus massless
  tracer stars). Different science, different scale, different goals. It uses
  Plummer softening, which is correct there and deliberately *absent* from the
  planetary N-body engine in `invisible_planet/`.

Run with: `uv run --python 3.12 --with taichi python prototypes/galaxy_collision_prototype.py`
