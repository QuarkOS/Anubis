"""
Anubis Avatar — Floating Cloud Sphere Overlay
==============================================
Entry point.  Creates a frameless, click-through, always-on-top window
and renders an animated cloud sphere inside it.

Usage:
    python -m shader_test.avatar          (from project root)
    python shader_test/avatar.py          (also works)
    Press Ctrl+C in the terminal to close.
"""

from __future__ import annotations

import time as _time

import moderngl
import pyglet

try:
    from shader_test.shaders import CLOUD_SPHERE
    from shader_test.renderer import QuadRenderer
    from shader_test.win32_overlay import setup_overlay, force_topmost
except ModuleNotFoundError:
    from shaders import CLOUD_SPHERE
    from renderer import QuadRenderer
    from win32_overlay import setup_overlay, force_topmost


class AvatarOverlay:
    """Transparent, click-through overlay that renders an animated cloud sphere."""

    SIZE = 300                     # px  (square window)
    TOPMOST_INTERVAL = 1.0        # seconds between re-asserting topmost

    def __init__(self):
        self._create_window()
        self._apply_overlay()
        self._init_renderer()
        self._wire_events()
        self._t0 = _time.perf_counter()

    # ── Window ────────────────────────────────────────────────

    def _create_window(self):
        cfg = pyglet.gl.Config(
            double_buffer=True,
            major_version=3, minor_version=3,
            alpha_size=8,
        )
        self._win = pyglet.window.Window(
            width=self.SIZE, height=self.SIZE,
            caption="Anubis",
            config=cfg,
            style=pyglet.window.Window.WINDOW_STYLE_TRANSPARENT,
        )
        # Bottom-right corner
        scr = self._win.screen
        self._win.set_location(scr.width - self.SIZE - 40,
                               scr.height - self.SIZE - 60)

    # ── Win32 overlay flags ───────────────────────────────────

    def _apply_overlay(self):
        self._hwnd = self._win._hwnd
        setup_overlay(self._hwnd)
        pyglet.clock.schedule_interval(self._tick_topmost, self.TOPMOST_INTERVAL)

    def _tick_topmost(self, _dt: float = 0) -> None:
        force_topmost(self._hwnd)

    # ── Renderer ──────────────────────────────────────────────

    def _init_renderer(self):
        self._ctx = moderngl.create_context()
        self._quad = QuadRenderer(self._ctx, CLOUD_SPHERE)

    # ── Events ────────────────────────────────────────────────

    def _wire_events(self):
        @self._win.event
        def on_draw():
            self._quad.render(
                _time.perf_counter() - self._t0,
                (float(self.SIZE), float(self.SIZE)),
            )

    # ── Run ───────────────────────────────────────────────────

    def run(self):
        pyglet.app.run()


if __name__ == '__main__':
    print("Anubis Avatar — floating cloud sphere overlay")
    print("Press Ctrl+C in this terminal to close.\n")
    AvatarOverlay().run()
