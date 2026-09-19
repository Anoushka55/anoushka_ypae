"""
Taichi backend initialisation.

Taichi must be initialised exactly once per process, before any field is
allocated, and re-initialising destroys every existing field. This module
provides the single idempotent entry point.

PRECISION POLICY (see docs/ARCHITECTURE.md §3.1)
------------------------------------------------
Physics runs in f64, always. This is not negotiable: in f32 the representable
time resolution at t ~ 1e4 days is ~86 s, which is the same order as the
transit-timing signals this project measures. f32 would manufacture numerical
"TTVs" indistinguishable from physical ones.

The CPU backend is the default because it is guaranteed to support f64. Taichi's
GPU backends do not all support f64 — in particular Vulkan (the GPU backend
available on the development machine) generally does not. Since the N-body
system has ~10 bodies, per-body parallelism is worthless anyway; the parallelism
that matters is across ENSEMBLE MEMBERS during parameter search, and the CPU
backend threads that dimension perfectly well.
"""

from __future__ import annotations

import taichi as ti

_INITIALISED = False
_ARCH_USED: str | None = None


def init_taichi(arch: str = "cpu", quiet: bool = True) -> str:
    """Initialise Taichi in f64. Idempotent; safe to call from anywhere.

    Returns the name of the architecture actually in use.
    """
    global _INITIALISED, _ARCH_USED
    if _INITIALISED:
        return _ARCH_USED  # type: ignore[return-value]

    arch_map = {"cpu": ti.cpu, "gpu": ti.gpu, "cuda": ti.cuda}
    requested = arch_map.get(arch, ti.cpu)

    kwargs = dict(default_fp=ti.f64, default_ip=ti.i32)
    if quiet:
        kwargs["log_level"] = ti.WARN

    try:
        ti.init(arch=requested, **kwargs)
        _ARCH_USED = arch
    except Exception:
        ti.init(arch=ti.cpu, **kwargs)
        _ARCH_USED = "cpu"

    _INITIALISED = True
    return _ARCH_USED


def is_initialised() -> bool:
    return _INITIALISED
