import threading
import tkinter as tk
import time
import win32con
import win32gui

class VisualFeedbackService:
    """
    Manages a soft, non-intrusive 'Corner Aura' for assistant state feedback.
    
    Replaces jarring flashes with a pulsing golden-cyan orb in the bottom-right
    corner to signal when Anubis is listening, thinking, or speaking.
    """

    def __init__(self):
        self.root = None
        self.canvas = None
        self._visible = False
        self._pulse_thread = None
        self._stop_pulse = threading.Event()
        self._current_state = "idle"
        
        # Colors (Anubis Gold & Cyan)
        self.COLOR_WAKING = "#FFD700"  # Gold
        self.COLOR_THINKING = "#00FFFF" # Cyan
        
        self._thread = threading.Thread(target=self._run_overlay, daemon=True)
        self._thread.start()

    def _run_overlay(self):
        """Initialize the corner overlay with click-through Win32 properties."""
        self.root = tk.Tk()
        self.root.title("AnubisAura")
        
        # Orb Size and Position (Bottom Right)
        size = 120
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        # Offset from corner
        x = screen_w - size - 20
        y = screen_h - size - 60
        
        self.root.geometry(f"{size}x{size}+{x}+{y}")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "black")
        self.root.config(bg="black")

        self.canvas = tk.Canvas(self.root, width=size, height=size, bg="black", highlightthickness=0)
        self.canvas.pack()

        # Win32 click-through
        hwnd = win32gui.FindWindow(None, "AnubisAura")
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, 
                               ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | 
                               win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW)

        self.root.mainloop()

    def set_state(self, state: str):
        """Update the aura state: 'idle', 'waking', 'thinking'."""
        if self._current_state == state:
            return
            
        self._current_state = state
        self._stop_pulse.set() # Kill existing animation
        
        if state == "idle":
            if self.canvas: self.canvas.delete("all")
            return

        # Start new pulse animation
        self._stop_pulse.clear()
        color = self.COLOR_WAKING if state == "waking" else self.COLOR_THINKING
        threading.Thread(target=self._animate_pulse, args=(color,), daemon=True).start()

    def _animate_pulse(self, color: str):
        """Pulse a soft-edged orb in the corner."""
        if not self.canvas: return
        
        center = 60
        max_r = 40
        min_r = 25
        
        while not self._stop_pulse.is_set():
            # Expansion
            for r in range(min_r, max_r):
                if self._stop_pulse.is_set(): break
                self._draw_orb(center, r, color)
                time.sleep(0.02)
            # Contraction
            for r in range(max_r, min_r, -1):
                if self._stop_pulse.is_set(): break
                self._draw_orb(center, r, color)
                time.sleep(0.03)

    def _draw_orb(self, center, radius, color):
        """Draw a multi-layered orb for a 'glow' effect."""
        self.canvas.delete("aura")
        # Glow layers
        for i in range(3, 0, -1):
            alpha_r = radius + (i * 8)
            # Tkinter doesn't do real alpha, so we simulate with color layers
            # or just a few rings for a 'halo' effect.
            self.canvas.create_oval(center-alpha_r, center-alpha_r, 
                                    center+alpha_r, center+alpha_r, 
                                    outline=color, width=2, tags="aura")
        # Core
        self.canvas.create_oval(center-radius, center-radius, 
                                center+radius, center+radius, 
                                fill=color, outline="", tags="aura")
