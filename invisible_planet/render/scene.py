"""
Visualisation of the Kepler-90 experiment.

Phase 13 deliverable.

=============================================================================
RENDERING CONSUMES STATE. IT NEVER DETERMINES IT.
=============================================================================
Nothing in this module writes to the simulation. The physics timestep is fixed
and comes from the convergence study; it is never tied to the frame rate. No
visual effect implies a force that is absent from the equations of motion, and
no orbit drawn here is a scripted ellipse — every point is a position the
integrator actually produced.

Planet X is drawn only when the narrative stage permits it. During the mystery
stage it is integrated (it must be, it is part of the physics) but not drawn,
exactly as an unseen planet is not seen.
=============================================================================
"""

# NOTE: no `from __future__ import annotations` — this module builds Taichi
# kernels, and PEP 563 string annotations break Taichi's signature parsing.

import numpy as np
import taichi as ti

from ..constants import R_SUN_IN_AU

# Colours chosen to stay legible on a near-black field, with the star warm and
# the planets cool so the hidden body reads as clearly different when revealed.
STAR_COLOUR = (1.0, 0.85, 0.45)
PLANET_COLOURS = [
    (0.55, 0.78, 1.00),
    (0.45, 0.90, 0.85),
    (0.70, 0.75, 1.00),
    (0.50, 0.85, 0.95),
    (0.60, 0.82, 1.00),
    (0.40, 0.75, 0.95),
    (0.65, 0.88, 0.90),
    (0.50, 0.70, 1.00),
]
PLANET_X_COLOUR = (1.00, 0.35, 0.40)
TRAIL_LENGTH = 900


class OrbitTrails:
    """Ring buffer of past positions, for drawing orbit trails.

    The trail is a record of where the integrator put each body. It is not a
    fitted or idealised ellipse.
    """

    def __init__(self, n_bodies: int, length: int = TRAIL_LENGTH):
        self.n = n_bodies
        self.length = length
        self.positions = ti.Vector.field(3, ti.f32, shape=(n_bodies, length))
        self.count = 0
        self.head = 0
        self._segments = [
            ti.Vector.field(3, ti.f32, shape=2 * (length - 1)) for _ in range(n_bodies)
        ]
        self._buffer = np.zeros((n_bodies, length, 3), dtype=np.float32)

    def push(self, positions: np.ndarray) -> None:
        self._buffer[:, self.head, :] = positions[: self.n].astype(np.float32)
        self.head = (self.head + 1) % self.length
        self.count = min(self.count + 1, self.length)

    def segment_field(self, body: int):
        """Vertex pairs for this body's trail, oldest to newest."""
        if self.count < 2:
            return None, 0
        order = [(self.head - 1 - k) % self.length for k in range(self.count)]
        points = self._buffer[body, order[::-1], :]
        pairs = np.empty((2 * (len(points) - 1), 3), dtype=np.float32)
        pairs[0::2] = points[:-1]
        pairs[1::2] = points[1:]
        field = self._segments[body]
        padded = np.zeros((field.shape[0], 3), dtype=np.float32)
        n = min(len(pairs), field.shape[0])
        padded[:n] = pairs[:n]
        field.from_numpy(padded)
        return field, n


class SystemRenderer:
    """Draws the star, the planets, their trails, and (when allowed) Planet X."""

    def __init__(self, n_bodies: int, window_size=(1400, 880)):
        self.n_bodies = n_bodies
        self.window = ti.ui.Window(
            "The Invisible Planet - Kepler-90 experiment", window_size, vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()

        self.body_positions = ti.Vector.field(3, ti.f32, shape=n_bodies)
        self.body_colours = ti.Vector.field(3, ti.f32, shape=n_bodies)
        self.star_position = ti.Vector.field(3, ti.f32, shape=1)
        self.trails = OrbitTrails(n_bodies)

        # camera state (simple orbit controller)
        self.azimuth = 0.6
        self.elevation = 0.85
        self.distance = 2.4
        self._last_mouse = None

        self._visible = np.ones(n_bodies, dtype=bool)

    # ------------------------------------------------------------------
    def set_visibility(self, index: int, visible: bool) -> None:
        """Hide a body from the DRAWING only; it still gravitates."""
        self._visible[index] = visible

    def handle_camera(self) -> None:
        mouse = np.array(self.window.get_cursor_pos())
        if self._last_mouse is None:
            self._last_mouse = mouse
        dx, dy = mouse - self._last_mouse
        if self.window.is_pressed(ti.ui.LMB):
            self.azimuth -= dx * 3.0
            self.elevation = float(np.clip(self.elevation + dy * 3.0, -1.5, 1.5))
        if self.window.is_pressed("w"):
            self.distance *= 0.98
        if self.window.is_pressed("s"):
            self.distance *= 1.02
        self.distance = float(np.clip(self.distance, 0.35, 12.0))
        self._last_mouse = mouse

    def draw(self, positions: np.ndarray, radii: np.ndarray, star_radius_au: float,
             reveal_planet_x: bool = True) -> None:
        """Render one frame from the current simulation state."""
        self.handle_camera()

        display = positions.astype(np.float32).copy()
        colours = np.zeros((self.n_bodies, 3), dtype=np.float32)
        colours[0] = STAR_COLOUR
        for i in range(1, self.n_bodies):
            colours[i] = PLANET_COLOURS[(i - 1) % len(PLANET_COLOURS)]
        # the last body is Planet X whenever the system carries one
        if self.n_bodies > 9:
            colours[-1] = PLANET_X_COLOUR if reveal_planet_x else (0.0, 0.0, 0.0)

        hidden = ~self._visible
        if self.n_bodies > 9 and not reveal_planet_x:
            hidden = hidden.copy()
            hidden[-1] = True
        # Bodies that must not be seen are moved far outside the view rather
        # than drawn in black, so they cannot occlude anything.
        display[hidden] = np.array([0.0, 0.0, 1e6], dtype=np.float32)

        self.body_positions.from_numpy(display)
        self.body_colours.from_numpy(colours)
        self.star_position.from_numpy(display[:1])

        centre = positions[0]
        cos_e = np.cos(self.elevation)
        eye = centre + self.distance * np.array([
            cos_e * np.cos(self.azimuth), cos_e * np.sin(self.azimuth),
            np.sin(self.elevation),
        ])
        self.camera.position(*eye)
        self.camera.lookat(*centre)
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.fov(45.0)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.85, 0.85, 0.9))
        self.scene.point_light(pos=tuple(float(v) for v in centre), color=(0.6, 0.6, 0.6))

        # orbit trails
        for body in range(1, self.n_bodies):
            if not self._visible[body]:
                continue
            if self.n_bodies > 9 and body == self.n_bodies - 1 and not reveal_planet_x:
                continue
            field, count = self.trails.segment_field(body)
            if field is not None and count > 0:
                colour = tuple(float(c) for c in colours[body] * 0.45)
                self.scene.lines(field, color=colour, width=1.0)

        # the star, drawn at a visible but honest scale
        self.scene.particles(
            self.star_position, radius=max(star_radius_au * 6.0, 0.012),
            color=STAR_COLOUR,
        )
        self.scene.particles(
            self.body_positions, radius=0.006, per_vertex_color=self.body_colours
        )

        self.canvas.set_background_color((0.015, 0.015, 0.025))
        self.canvas.scene(self.scene)

    def running(self) -> bool:
        return self.window.running

    def show(self) -> None:
        self.window.show()
