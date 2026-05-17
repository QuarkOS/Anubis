import sys
import threading
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QRadialGradient

class AuraSignal(QObject):
    state_changed = pyqtSignal(str)

class AuraWidget(QWidget):
    """
    Hardware-accelerated PyQt6 Compositor UI.
    Provides a 60FPS transparent glow utilizing the Windows DWM.
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
        self.size = 200
        self.resize(self.size, self.size)
        
        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - self.size - 20
        y = screen.height() - self.size - 60
        self.move(x, y)
        
        self.current_state = "idle"
        self.color = QColor(0, 0, 0, 0)
        
        # 60 FPS Animation loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)
        
        self.pulse_val = 0
        self.pulse_dir = 1

    def set_state(self, state: str):
        self.current_state = state
        if state == "idle":
            self.color = QColor(0, 0, 0, 0)
            self.hide()
        else:
            if state == "waking":
                self.color = QColor(255, 215, 0, 180) # Gold
            elif state == "thinking":
                self.color = QColor(0, 255, 255, 180) # Cyan
            self.show()

    def update_animation(self):
        if self.current_state == "idle":
            return
            
        self.pulse_val += 1.5 * self.pulse_dir
        if self.pulse_val >= 30:
            self.pulse_dir = -1
        elif self.pulse_val <= 0:
            self.pulse_dir = 1
            
        self.update() # Schedule paintEvent

    def paintEvent(self, event):
        if self.current_state == "idle":
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = self.rect().center()
        base_radius = 40
        current_radius = base_radius + self.pulse_val
        
        # Radial Gradient for smooth glow/blur effect calculated on GPU/DWM
        gradient = QRadialGradient(center, current_radius)
        gradient.setColorAt(0, self.color)
        gradient.setColorAt(0.7, QColor(self.color.red(), self.color.green(), self.color.blue(), 50))
        gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, int(current_radius), int(current_radius))


class VisualFeedbackService:
    """
    Manages the Qt Event Loop in a dedicated background thread to prevent blocking
    the main asyncio orchestration loop.
    """

    def __init__(self):
        self.app = None
        self.widget = None
        self.signals = AuraSignal()
        
        self._thread = threading.Thread(target=self._run_qt_app, daemon=True)
        self._thread.start()

    def _run_qt_app(self):
        self.app = QApplication(sys.argv)
        self.widget = AuraWidget()
        
        # Safely marshal thread signals into the Qt event loop
        self.signals.state_changed.connect(self.widget.set_state)
        
        self.app.exec()

    def set_state(self, state: str):
        """Valid states: 'idle', 'waking', 'thinking'"""
        self.signals.state_changed.emit(state)

    def trigger_wake_pulse(self):
        self.set_state("waking")

    def show_thinking_glow(self):
        self.set_state("thinking")

    def hide_glow(self):
        self.set_state("idle")
