"""
Win32 overlay helpers — frameless, click-through, always-on-top windows.

Uses ctypes with properly declared argtypes so that HWND_TOPMOST (-1)
is marshalled as a pointer-width signed value, which is the root cause
of SetWindowPos silently failing on 64-bit Python.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt

# ── Typed Win32 function references ──────────────────────────

_user32 = ctypes.windll.user32

_SetWindowLongW = _user32.SetWindowLongW
_SetWindowLongW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_long]
_SetWindowLongW.restype = ctypes.c_long

_GetWindowLongW = _user32.GetWindowLongW
_GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
_GetWindowLongW.restype = ctypes.c_long

_SetWindowPos = _user32.SetWindowPos
_SetWindowPos.argtypes = [
    wt.HWND,   # hWnd
    wt.HWND,   # hWndInsertAfter  ← must be pointer-width for -1
    ctypes.c_int, ctypes.c_int,   # X, Y
    ctypes.c_int, ctypes.c_int,   # cx, cy
    wt.UINT,                      # uFlags
]
_SetWindowPos.restype = wt.BOOL


# ── Constants ────────────────────────────────────────────────

GWL_STYLE   = -16
GWL_EXSTYLE = -20

WS_POPUP   = 0x80000000
WS_VISIBLE = 0x10000000

WS_EX_TOPMOST     = 0x00000008
WS_EX_TRANSPARENT  = 0x00000020   # click-through
WS_EX_LAYERED      = 0x00080000
WS_EX_TOOLWINDOW   = 0x00000080   # hide from taskbar

# HWND_TOPMOST must be cast to a proper HWND (pointer) so that
# the value -1 sign-extends correctly on 64-bit Windows.
HWND_TOPMOST = wt.HWND(-1)

SWP_NOMOVE       = 0x0002
SWP_NOSIZE       = 0x0001
SWP_NOACTIVATE   = 0x0010
SWP_FRAMECHANGED = 0x0020


# ── Public API ───────────────────────────────────────────────

def make_frameless(hwnd: int) -> None:
    """Strip the title bar and all borders from *hwnd*."""
    _SetWindowLongW(wt.HWND(hwnd), GWL_STYLE, WS_POPUP | WS_VISIBLE)


def make_click_through(hwnd: int) -> None:
    """Make *hwnd* completely ignore mouse input (pass-through)."""
    ex = _GetWindowLongW(wt.HWND(hwnd), GWL_EXSTYLE)
    _SetWindowLongW(
        wt.HWND(hwnd), GWL_EXSTYLE,
        ex | WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW,
    )


def force_topmost(hwnd: int) -> None:
    """Pin *hwnd* above every other window and commit pending style changes."""
    _SetWindowPos(
        wt.HWND(hwnd), HWND_TOPMOST,
        0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
    )


def setup_overlay(hwnd: int) -> None:
    """One-call convenience: frameless + click-through + topmost."""
    make_frameless(hwnd)
    make_click_through(hwnd)
    force_topmost(hwnd)
