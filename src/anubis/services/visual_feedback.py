import sys
import threading
import time
import math
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF
from PyQt6.QtGui import QPainter, QColor, QRadialGradient

class AuraSignal(QObject):
    state_changed = pyqtSignal(str)

class AuraWidget(QWidget):
    """
    Hardware-accelerated PyQt6 Compositor UI with premium volumetric-like blending.
    Provides a 60FPS transparent glowing plasma orb utilizing the Windows DWM.
    """
    def __init__(self):
        super().__init__()
        # Set flags for a completely transparent, click-through, always-on-top overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Geometry: Bottom Right Corner
        self.size = 240
        self.resize(self.size, self.size)
        
        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - self.size - 20
        y = screen.height() - self.size - 60
        self.move(x, y)
        
        self.current_state = "idle"
        self.base_color = QColor(0, 0, 0, 0)
        self.core_color = QColor(0, 0, 0, 0)
        
        # 60 FPS Animation loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)
        
        self.start_time = time.time()
        self.pulse_val = 0
        self.pulse_dir = 1

    def set_state(self, state: str):
        self.current_state = state
        if state == "idle":
            self.hide()
        else:
            if state == "waking":
                self.base_color = QColor(255, 140, 0, 110)   # Deep Amber
                self.core_color = QColor(255, 220, 100, 200) # Bright Gold
            elif state == "thinking":
                self.base_color = QColor(0, 150, 255, 110)   # Deep Blue
                self.core_color = QColor(100, 255, 255, 200) # Bright Cyan
            self.show()

    def update_animation(self):
        if self.current_state == "idle":
            return
            
        self.pulse_val += 1.0 * self.pulse_dir
        if self.pulse_val >= 20:
            self.pulse_dir = -1
        elif self.pulse_val <= 0:
            self.pulse_dir = 1
            
        self.update() # Schedule paintEvent

    def paintEvent(self, event):
        if self.current_state == "idle":
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)
        
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        base_radius = 50 + (self.pulse_val * 0.5)
        
        t = (time.time() - self.start_time) * 2.0
        
        # Draw 3 overlapping, orbiting gradient spheres to simulate a volumetric plasma core
        for i in range(3):
            offset_t = t + (i * 2.09) # ~120 degree phase shift
            orb_x = center_x + math.sin(offset_t) * 12
            orb_y = center_y + math.cos(offset_t * 1.3) * 12
            orb_radius = base_radius + math.sin(t * 1.5 + i) * 10
            
            gradient = QRadialGradient(QPointF(orb_x, orb_y), float(orb_radius))
            gradient.setColorAt(0, self.core_color)
            gradient.setColorAt(0.4, self.base_color)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            
            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(orb_x, orb_y), float(orb_radius), float(orb_radius))


class VisualFeedbackService(QObject):
    """
    Manages the Qt Widget. Must be instantiated in the main thread.
    Exposes thread-safe methods to update the state from background threads.
    """
    state_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.widget = AuraWidget()
        # Safely marshal thread signals into the Qt event loop
        self.state_changed.connect(self.widget.set_state)

    def set_state(self, state: str):
        """Valid states: 'idle', 'waking', 'thinking'"""
        self.state_changed.emit(state)

    def trigger_wake_pulse(self):
        self.set_state("waking")

    def show_thinking_glow(self):
        self.set_state("thinking")

    def hide_glow(self):
        self.set_state("idle")
