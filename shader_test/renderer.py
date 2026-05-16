"""
ModernGL fullscreen-quad renderer for Shadertoy-style fragment shaders.
"""

from __future__ import annotations

import struct

import moderngl

try:
    from shader_test.shaders import VERTEX_SHADER, FRAGMENT_WRAPPER
except ImportError:
    from shaders import VERTEX_SHADER, FRAGMENT_WRAPPER


class QuadRenderer:
    """Compiles a Shadertoy-style fragment shader and renders it on a quad."""

    def __init__(self, ctx: moderngl.Context, shader_body: str):
        self._ctx = ctx

        # Full-screen triangle-strip quad: 4 vertices, 2 floats each
        vbo = ctx.buffer(struct.pack(
            '8f',
            -1, -1,  1, -1,
            -1,  1,  1,  1,
        ))

        frag_src = FRAGMENT_WRAPPER.format(shader_body=shader_body)
        self._prog = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=frag_src,
        )
        self._vao = ctx.vertex_array(self._prog, [(vbo, '2f', 'in_position')])

    def render(self, time: float, resolution: tuple[float, float]) -> None:
        """Draw one frame.  *resolution* is (width, height) in pixels."""
        ctx = self._ctx
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        prog = self._prog
        if 'iTime' in prog:
            prog['iTime'].value = time
        if 'iResolution' in prog:
            prog['iResolution'].value = resolution
        if 'iMouse' in prog:
            prog['iMouse'].value = (0.0, 0.0)

        self._vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
