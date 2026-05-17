import threading
import tkinter as tk
import time
import win32api
import win32con
import win32gui

class VisualFeedbackService:
    """
    Manages a transparent, click-through overlay for real-time visual feedback.
    
    Provides 'Spectral Glow' effects to signal assistant state (Waking, Thinking, Speaking)
    without interrupting the user's workflow or stealing window focus.
    """

    def __init__(self):
        self.root = None
        self.canvas = None
        self._thread = threading.Thread(target=self._run_overlay, daemon=True)
        self._visible = False
        self._pulse_active = False
        self._thread.start()

    def _run_overlay(self):
        """Initialize the transparent tkinter window and Win32 click-through properties."""
        self.root = tk.Tk()
        self.root.title("AnubisOverlay")
        
        # Geometry: Full screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Transparency and Z-Order
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "black")
        self.root.config(bg="black")

        self.canvas = tk.Canvas(self.root, width=screen_width, height=screen_height, bg="black", highlightthickness=0)
        self.canvas.pack()

        # Win32 magic to make it click-through and truly transparent to events
        hwnd = win32gui.FindWindow(None, "AnubisOverlay")
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        # WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, 
                               ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | 
                               win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW)

        self.root.mainloop()

    def trigger_wake_pulse(self):
        """Flash a golden-cyan border pulse across the screen edges."""
        if not self.canvas: return
        threading.Thread(target=self._animate_pulse, daemon=True).start()

    def show_thinking_glow(self):
        """Show a persistent subtle glow at the bottom of the screen."""
        self._visible = True
        self._draw_state("thinking")

    def hide_glow(self):
        """Remove all active overlays."""
        self._visible = False
        if self.canvas:
            self.canvas.delete("all")

    def _animate_pulse(self):
        """Run the pulse animation frame-by-frame on the tkinter canvas."""
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        
        # Colors: Anubis Gold/Cyan
        colors = ["#FFD700", "#00FFFF", "#FFD700"]
        
        for i in range(1, 15):
            thickness = i * 2
            self.canvas.delete("pulse")
            # Draw rectangles for the pulse effect
            self.canvas.create_rectangle(0, 0, w, h, outline=colors[i % 2], width=thickness, tags="pulse")
            time.sleep(0.02)
            
        for i in range(15, 0, -1):
            thickness = i * 2
            self.canvas.delete("pulse")
            self.canvas.create_rectangle(0, 0, w, h, outline="#FFD700", width=thickness, tags="pulse")
            time.sleep(0.02)
            
        self.canvas.delete("pulse")

    def _draw_state(self, state: str):
        """Update the canvas with state-specific visual cues."""
        if not self.canvas: return
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        self.canvas.delete("state")

        if state == "thinking":
            # Subtle golden bar at bottom
            self.canvas.create_rectangle(w//4, h-5, 3*w//4, h, fill="#FFD700", outline="", tags="state")
