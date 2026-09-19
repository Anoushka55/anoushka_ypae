"""
The 3-D view of the Kepler-90 experiment.

=============================================================================
RENDERING CONSUMES STATE. IT NEVER DETERMINES IT.
=============================================================================
Nothing in this module writes to the simulation. The physics timestep is fixed by the
convergence study and is never tied to the frame rate. No orbit drawn here is a scripted ellipse:
every trail point is a position the integrator produced. Velocity arrows show the integrator's
velocities; perturbation arrows show one term of the Newtonian force sum (the pull of the hidden
planet alone), computed by the engine. Nothing drawn implies a force absent from the equations.

The hidden planet is drawn only when the caller says so (`reveal_hidden`). Before that it is
integrated - it must be, its gravity is what bends the transit times - but not drawn, exactly as
an unseen planet is not seen.

SIZES ARE NOT TO SCALE. Bodies are drawn far larger than life so that they are visible: the star
is drawn at 6x its radius, and planets at a fixed size unrelated to their radii. The UI says so.
=============================================================================
"""

# NOTE: no `from __future__ import annotations` - this module builds Taichi fields.

import numpy as np
import taichi as ti

STAR_COLOUR = (1.0, 0.85, 0.45)
PLANET_COLOURS = [
    (0.55, 0.78, 1.00), (0.45, 0.90, 0.85), (0.70, 0.75, 1.00), (0.50, 0.85, 0.95),
    (0.60, 0.82, 1.00), (0.40, 0.75, 0.95), (0.65, 0.88, 0.90), (0.50, 0.70, 1.00),
]
HIDDEN_COLOUR = (1.00, 0.35, 0.40)
TRAIL_LENGTH = 700
GHOST_COLOUR = (1.00, 0.80, 0.20)       # the CANDIDATE Planet X reconstructed by the inference
VELOCITY_COLOUR = (0.35, 0.95, 0.45)
PERTURBATION_COLOUR = (1.00, 0.55, 0.20)


class OrbitTrails:
    """Ring buffer of past positions. The trail is a record of where the integrator put each
    body, not a fitted or idealised ellipse."""

    def __init__(self, n_bodies: int, length: int = TRAIL_LENGTH):
        self.n, self.length = n_bodies, length
        self.head, self.count = 0, 0
        self._buffer = np.zeros((n_bodies, length, 3), dtype=np.float32)
        self._segments = [ti.Vector.field(3, ti.f32, shape=2 * (length - 1)) for _ in range(n_bodies)]

    def push(self, positions: np.ndarray) -> None:
        self._buffer[:, self.head, :] = positions[: self.n].astype(np.float32)
        self.head = (self.head + 1) % self.length
        self.count = min(self.count + 1, self.length)

    def clear(self) -> None:
        self.head = self.count = 0

    def field_for(self, body: int):
        if self.count < 2:
            return None
        order = [(self.head - 1 - k) % self.length for k in range(self.count)][::-1]
        pts = self._buffer[body, order, :]
        pairs = np.empty((2 * (len(pts) - 1), 3), dtype=np.float32)
        pairs[0::2], pairs[1::2] = pts[:-1], pts[1:]
        field = self._segments[body]
        padded = np.full((field.shape[0], 3), 1e6, dtype=np.float32)
        padded[: len(pairs)] = pairs
        field.from_numpy(padded)
        return field


class SystemRenderer:
    """Draws the star, the planets, their trails, velocity and perturbation arrows."""

    def __init__(self, n_bodies: int, window_size=(1400, 880), show_window: bool = True,
                 title: str = "The Invisible Planet - Kepler-90 experiment"):
        self.n_bodies = n_bodies
        self.window = ti.ui.Window(title, window_size, vsync=False, show_window=show_window)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.trails = OrbitTrails(n_bodies)
        self.ghost_trails = OrbitTrails(1)
        self.ghost_position = ti.Vector.field(3, ti.f32, shape=1)

        self.body_positions = ti.Vector.field(3, ti.f32, shape=n_bodies)
        self.body_colours = ti.Vector.field(3, ti.f32, shape=n_bodies)
        self.star_position = ti.Vector.field(3, ti.f32, shape=1)
        self.velocity_lines = ti.Vector.field(3, ti.f32, shape=2 * n_bodies)
        self.perturbation_lines = ti.Vector.field(3, ti.f32, shape=2 * n_bodies)
        self.tip_positions = ti.Vector.field(3, ti.f32, shape=n_bodies)

        self.azimuth, self.elevation, self.distance = 0.6, 0.85, 2.4
        self.target = np.zeros(3)
        self._last_mouse = None
        # Camera frame aligned with the planets' orbital plane (a view choice only; the
        # simulation's coordinates are untouched). Kepler-90 is nearly edge-on to the sky
        # plane, so a fixed world-z "up" would show the orbits foreshortened.
        self._basis = np.eye(3)
        self._aligned = False
        self.capture_every_frame = False      # set True before show() if screenshots are wanted
        self._last_image = None

    # ---- camera ----------------------------------------------------------------------
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
        self.distance = float(np.clip(self.distance, 0.25, 12.0))
        self._last_mouse = mouse

    def align_to_orbital_plane(self, positions: np.ndarray, velocities: np.ndarray, skip=()) -> None:
        """Point the camera's 'up' along the total orbital angular momentum of the bodies about
        body 0. Purely a view transform."""
        keep = [i for i in range(1, len(positions)) if i not in skip]
        rel_p = positions[keep] - positions[0]
        rel_v = velocities[keep] - velocities[0]
        normal = np.cross(rel_p, rel_v).sum(axis=0)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            return
        normal /= norm
        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(helper, normal)
        e1 /= np.linalg.norm(e1)
        self._basis = np.stack([e1, np.cross(normal, e1), normal], axis=1)   # columns e1, e2, n

    def _set_camera(self, centre: np.ndarray) -> None:
        cos_e = np.cos(self.elevation)
        local = np.array([cos_e * np.cos(self.azimuth), cos_e * np.sin(self.azimuth), np.sin(self.elevation)])
        eye = centre + self.distance * (self._basis @ local)
        self.camera.position(*eye)
        self.camera.lookat(*centre)
        self.camera.up(*self._basis[:, 2])
        self.camera.fov(45.0)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.85, 0.85, 0.9))
        self.scene.point_light(pos=tuple(float(v) for v in centre), color=(0.6, 0.6, 0.6))

    # ---- drawing -----------------------------------------------------------------------
    def draw(self, positions: np.ndarray, star_radius_au: float, *, velocities=None,
             hidden_index=None, reveal_hidden=False, selected: int | None = None,
             show_trails=True, show_velocity=False, perturbation=None, flash=(),
             ghost=None, interactive_camera=True) -> None:
        """Render one frame from the current simulation state.

        positions      (N, 3) au
        velocities     (N, 3) au/day, needed only for velocity arrows
        perturbation   (N, 3) acceleration of each body due to the hidden planet alone, or None
        flash          indices of bodies that just transited (drawn brighter)
        ghost          (3,) position of the CANDIDATE Planet X from the inference, or None. The
                       caller pushes it to `ghost_trails` each frame.
        """
        if interactive_camera:
            self.handle_camera()
        if velocities is not None and not self._aligned:
            self.align_to_orbital_plane(positions, velocities, skip=() if hidden_index is None else (hidden_index,))
            self._aligned = True
        pos = positions.astype(np.float32)
        centre = pos[0] if self.target is None else pos[0] + self.target

        colours = np.zeros((self.n_bodies, 3), dtype=np.float32)
        colours[0] = STAR_COLOUR
        for i in range(1, self.n_bodies):
            colours[i] = PLANET_COLOURS[(i - 1) % len(PLANET_COLOURS)]
        for i in flash:
            colours[i] = (1.0, 1.0, 1.0)
        if selected is not None and 0 < selected < self.n_bodies:
            colours[selected] = np.clip(colours[selected] * 1.25 + 0.1, 0, 1)
        if hidden_index is not None:
            colours[hidden_index] = HIDDEN_COLOUR

        display = pos.copy()
        hide = np.zeros(self.n_bodies, dtype=bool)
        if hidden_index is not None and not reveal_hidden:
            hide[hidden_index] = True
        # bodies that must not be seen are moved far away rather than drawn black, so they
        # cannot occlude anything
        display[hide] = np.array([0.0, 0.0, 1e6], dtype=np.float32)

        self.body_positions.from_numpy(display)
        self.body_colours.from_numpy(colours)
        self.star_position.from_numpy(display[:1])
        self._set_camera(centre.astype(np.float64))

        if show_trails:
            for body in range(1, self.n_bodies):
                if hide[body]:
                    continue
                field = self.trails.field_for(body)
                if field is not None:
                    self.scene.lines(field, color=tuple(float(c) for c in colours[body] * 0.45), width=1.0)

        if show_velocity and velocities is not None:
            speeds = np.linalg.norm(velocities[1:], axis=1)
            scale = 0.10 / max(float(speeds.max()), 1e-12)          # longest arrow = 0.10 au
            verts = np.full((2 * self.n_bodies, 3), 1e6, dtype=np.float32)
            for i in range(1, self.n_bodies):
                if hide[i]:
                    continue
                verts[2 * i] = pos[i]
                verts[2 * i + 1] = pos[i] + (velocities[i] * scale).astype(np.float32)
            self.velocity_lines.from_numpy(verts)
            self.scene.lines(self.velocity_lines, color=VELOCITY_COLOUR, width=2.0)

        if perturbation is not None:
            mags = np.linalg.norm(perturbation, axis=1)
            scale = 0.08 / max(float(mags.max()), 1e-30)            # longest arrow = 0.08 au
            verts = np.full((2 * self.n_bodies, 3), 1e6, dtype=np.float32)
            for i in range(1, self.n_bodies):
                if hide[i] or i == hidden_index:
                    continue
                verts[2 * i] = pos[i]
                verts[2 * i + 1] = pos[i] + (perturbation[i] * scale).astype(np.float32)
            self.perturbation_lines.from_numpy(verts)
            self.scene.lines(self.perturbation_lines, color=PERTURBATION_COLOUR, width=3.0)

        if ghost is not None:
            self.ghost_position.from_numpy(np.asarray(ghost, dtype=np.float32).reshape(1, 3))
            field = self.ghost_trails.field_for(0)
            if field is not None:
                self.scene.lines(field, color=tuple(c * 0.5 for c in GHOST_COLOUR), width=1.0)
            self.scene.particles(self.ghost_position, radius=0.010, color=GHOST_COLOUR)

        self.scene.particles(self.star_position, radius=max(star_radius_au * 6.0, 0.012), color=STAR_COLOUR)
        self.scene.particles(self.body_positions, radius=0.006, per_vertex_color=self.body_colours)
        self.canvas.set_background_color((0.015, 0.015, 0.025))
        self.canvas.scene(self.scene)

    # ---- window --------------------------------------------------------------------------
    def running(self) -> bool:
        return self.window.running

    def show(self) -> None:
        self.window.show()
        if self.capture_every_frame:
            self._last_image = self.window.get_image_buffer_as_numpy()

    def screenshot(self, path: str) -> None:
        """Save the current frame as a PNG (call after show())."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # GGUI fills the image buffer only if it is requested on every frame, hence
        # `capture_every_frame` (a read-only copy; it does not affect the simulation).
        buffer = self._last_image if self._last_image is not None else self.window.get_image_buffer_as_numpy()
        # (width, height, 4), origin bottom-left
        plt.imsave(path, np.transpose(buffer, (1, 0, 2))[::-1])
