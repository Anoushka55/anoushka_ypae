"""
Entry point.

  python -m invisible_planet.app.main            autonomous demonstration
  python -m invisible_planet.app.main --speed 2  faster playback
  python -m invisible_planet.app.main --investigate   interactive mode

`--speed` changes how many fixed-size physics steps are taken per frame. It
never changes the timestep itself, which is fixed by the convergence study.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="The Invisible Planet")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="physics steps per frame multiplier (not the timestep)")
    parser.add_argument("--investigate", action="store_true",
                        help="interactive investigation mode")
    args = parser.parse_args()

    if args.investigate:
        from .investigate import run
        run()
    else:
        from .story import run
        run(speed=args.speed)


if __name__ == "__main__":
    main()
