"""
2-D scientific plots drawn on the GGUI canvas: O-C diagrams, periodograms, chi2 profiles.

GGUI has no text drawing on the canvas, so titles and axis labels are written with the GUI
text widgets by the caller; this module draws frames, grid lines, points and polylines.

Coordinates are normalised canvas coordinates in [0, 1] with the origin at the bottom left.
A `Box` is a rectangle of the canvas; `to_box` maps data coordinates into it. All arrays are
handed to Taichi fields of fixed size (a draw call always draws the whole field), unused
entries being parked off-screen.

Nothing here computes physics: it draws arrays it is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import taichi as ti

OFFSCREEN = -10.0
MAX_POINTS = 1024
MAX_SEGMENTS = 1024


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_box(self, x, y, xlim, ylim) -> np.ndarray:
        """Map data coordinates into this box; returns (n, 2) float32."""
        x = (np.asarray(x, dtype=np.float64) - xlim[0]) / (xlim[1] - xlim[0])
        y = (np.asarray(y, dtype=np.float64) - ylim[0]) / (ylim[1] - ylim[0])
        return np.stack([self.x0 + x * (self.x1 - self.x0), self.y0 + y * (self.y1 - self.y0)], axis=1).astype(np.float32)


class PlotPool:
    """Preallocated fields for one plot layer."""

    def __init__(self):
        self.points = ti.Vector.field(2, ti.f32, shape=MAX_POINTS)
        self.point_colors = ti.Vector.field(3, ti.f32, shape=MAX_POINTS)
        self.vertices = ti.Vector.field(2, ti.f32, shape=2 * MAX_SEGMENTS)
        self.line_colors = ti.Vector.field(3, ti.f32, shape=2 * MAX_SEGMENTS)


def _clip(xy: np.ndarray, box: Box) -> np.ndarray:
    inside = (xy[:, 0] >= box.x0) & (xy[:, 0] <= box.x1) & (xy[:, 1] >= box.y0) & (xy[:, 1] <= box.y1)
    out = xy.copy()
    out[~inside] = OFFSCREEN
    return out


def draw_points(canvas, pool: PlotPool, xy: np.ndarray, color, radius: float, box: Box | None = None) -> None:
    n = min(len(xy), MAX_POINTS)
    buffer = np.full((MAX_POINTS, 2), OFFSCREEN, dtype=np.float32)
    buffer[:n] = _clip(xy[:n], box) if box is not None else xy[:n]
    colors = np.tile(np.asarray(color, dtype=np.float32), (MAX_POINTS, 1))
    pool.points.from_numpy(buffer)
    pool.point_colors.from_numpy(colors)
    canvas.circles(pool.points, radius=radius, per_vertex_color=pool.point_colors)


def draw_polyline(canvas, pool: PlotPool, xy: np.ndarray, color, width: float = 0.002, box: Box | None = None) -> None:
    n = min(len(xy), MAX_SEGMENTS + 1)
    buffer = np.full((2 * MAX_SEGMENTS, 2), OFFSCREEN, dtype=np.float32)
    if n >= 2:
        pts = _clip(xy[:n], box) if box is not None else xy[:n]
        buffer[0 : 2 * (n - 1) : 2] = pts[:-1]
        buffer[1 : 2 * (n - 1) : 2] = pts[1:]
    colors = np.tile(np.asarray(color, dtype=np.float32), (2 * MAX_SEGMENTS, 1))
    pool.vertices.from_numpy(buffer)
    pool.line_colors.from_numpy(colors)
    canvas.lines(pool.vertices, width=width, per_vertex_color=pool.line_colors)


def draw_segments(canvas, pool: PlotPool, pairs: np.ndarray, color, width: float = 0.0015,
                  box: Box | None = None) -> None:
    """Independent line segments: `pairs` is (n, 2, 2), each row being the two endpoints.
    Used for error bars and threshold lines."""
    n = min(len(pairs), MAX_SEGMENTS)
    buffer = np.full((2 * MAX_SEGMENTS, 2), OFFSCREEN, dtype=np.float32)
    if n:
        flat = pairs[:n].reshape(-1, 2).astype(np.float32)
        if box is not None:
            flat = np.clip(flat, [box.x0, box.y0], [box.x1, box.y1])
        buffer[: 2 * n] = flat
    colors = np.tile(np.asarray(color, dtype=np.float32), (2 * MAX_SEGMENTS, 1))
    pool.vertices.from_numpy(buffer)
    pool.line_colors.from_numpy(colors)
    canvas.lines(pool.vertices, width=width, per_vertex_color=pool.line_colors)


def draw_frame(canvas, pool: PlotPool, box: Box, color=(0.55, 0.6, 0.7), zero_line: float | None = None,
               ylim=None, n_grid: int = 4) -> None:
    """The plot frame, light grid lines and an optional zero line."""
    corners = np.array([[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1],
                        [box.x0, box.y0]], dtype=np.float32)
    draw_polyline(canvas, pool, corners, color, width=0.0015)


def draw_grid(canvas, pool: PlotPool, box: Box, n: int = 4, color=(0.18, 0.2, 0.26)) -> None:
    lines = []
    for k in range(1, n):
        f = k / n
        lines.append([[box.x0 + f * (box.x1 - box.x0), box.y0], [box.x0 + f * (box.x1 - box.x0), box.y1]])
        lines.append([[box.x0, box.y0 + f * (box.y1 - box.y0)], [box.x1, box.y0 + f * (box.y1 - box.y0)]])
    buffer = np.full((2 * MAX_SEGMENTS, 2), OFFSCREEN, dtype=np.float32)
    for i, (a, b) in enumerate(lines):
        buffer[2 * i], buffer[2 * i + 1] = a, b
    pool.vertices.from_numpy(buffer)
    pool.line_colors.from_numpy(np.tile(np.asarray(color, dtype=np.float32), (2 * MAX_SEGMENTS, 1)))
    canvas.lines(pool.vertices, width=0.001, per_vertex_color=pool.line_colors)


def nice_limits(values, pad: float = 0.1, minimum_span: float = 1e-9) -> tuple:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = max(hi - lo, minimum_span)
    return lo - pad * span, hi + pad * span
