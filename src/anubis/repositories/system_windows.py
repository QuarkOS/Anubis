import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import GPUtil
import psutil
import pyperclip
import requests
import win32gui
from pydantic import Field

from anubis.domain.protocols import SystemProbe
from anubis.domain.schemas import GPUInfo, NetworkState, ProcessInfo, SystemState, UserContext


class WindowsSystemProbe(SystemProbe):
    """Windows-specific implementation of the SystemProbe with rich telemetry."""

    def __init__(self):
        """Initialize the probe and capture initial network counters for delta calculations."""
        self._last_net_io = psutil.net_io_counters()
        self._last_net_time = time.time()
        self._cached_ip_info = None
        self._last_ip_check = 0

    async def probe_state(self) -> SystemState:
        """
        Capture a comprehensive snapshot of the Windows system state.
        
        This method aggregates telemetry from CPU, memory, battery, GPU, 
        network, and user context to provide a full situational profile.
        """
        # CPU and Memory
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else None
        
        # Active Window
        try:
            hwnd = win32gui.GetForegroundWindow()
            active_window = win32gui.GetWindowText(hwnd)
        except Exception:
            active_window = "Unknown"

        # 1. Top Processes
        top_processes = self._get_top_processes()

        # 2. GPU Info
        gpus = self._get_gpu_info()

        # 3. Network Info
        network = self._get_network_info()

        # 4. User Context
        user_context = self._get_user_context()

        return SystemState(
            cpu_percent=cpu,
            memory_percent=memory,
            battery_percent=battery_percent,
            active_window=active_window,
            top_processes=top_processes,
            gpus=gpus,
            network=network,
            user_context=user_context,
            os_name=f"{platform.system()} {platform.release()}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _get_top_processes(self) -> list[ProcessInfo]:
        """Identify and profile the top 5 resource-consuming processes by CPU usage."""
        top_processes = []
        try:
            procs = []
            for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    procs.append({
                        'pid': p.info['pid'],
                        'name': p.info['name'],
                        'cpu': p.info['cpu_percent'] or 0.0,
                        'mem': (p.info['memory_info'].rss / (1024 * 1024)) if p.info['memory_info'] else 0.0
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            sorted_by_cpu = sorted(procs, key=lambda x: x['cpu'], reverse=True)[:5]
            for p in sorted_by_cpu:
                top_processes.append(ProcessInfo(
                    name=p['name'] or "Unknown",
                    cpu_percent=p['cpu'],
                    memory_mb=p['mem'],
                    pid=p['pid']
                ))
        except Exception:
            pass
        return top_processes

    def _get_gpu_info(self) -> list[GPUInfo]:
        """Retrieve telemetry for all available NVIDIA GPUs using GPUtil."""
        gpu_list = []
        try:
            gpus = GPUtil.getGPUs()
            for g in gpus:
                gpu_list.append(GPUInfo(
                    name=g.name,
                    load_percent=g.load * 100,
                    memory_used_mb=g.memoryUsed,
                    memory_total_mb=g.memoryTotal,
                    temperature=g.temperature
                ))
        except Exception:
            pass
        return gpu_list

    def _get_network_info(self) -> NetworkState:
        """
        Calculate network throughput and gather connection metadata.
        
        Includes Wi-Fi SSID detection and cached public IP/location lookup.
        """
        # Network Speed
        current_io = psutil.net_io_counters()
        current_time = time.time()
        elapsed = current_time - self._last_net_time
        
        up_kbps = ((current_io.bytes_sent - self._last_net_io.bytes_sent) / 1024) / elapsed if elapsed > 0 else 0
        down_kbps = ((current_io.bytes_recv - self._last_net_io.bytes_recv) / 1024) / elapsed if elapsed > 0 else 0
        
        self._last_net_io = current_io
        self._last_net_time = current_time

        # SSID
        ssid = None
        try:
            out = subprocess.check_output("netsh wlan show interfaces", shell=True).decode('utf-8', errors='ignore')
            for line in out.split('\n'):
                if "SSID" in line and "BSSID" not in line:
                    ssid = line.split(":")[1].strip()
                    break
        except Exception:
            pass

        # Public IP & Location (Cached for 1 hour)
        if not self._cached_ip_info or (current_time - self._last_ip_check > 3600):
            try:
                # Use a timeout to not block too long
                resp = requests.get('https://ipapi.co/json/', timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    self._cached_ip_info = {
                        'ip': data.get('ip'),
                        'location': f"{data.get('city')}, {data.get('region')}, {data.get('country_name')}"
                    }
                    self._last_ip_check = current_time
            except Exception:
                pass

        return NetworkState(
            ssid=ssid,
            upload_kbps=up_kbps,
            download_kbps=down_kbps,
            public_ip=self._cached_ip_info.get('ip') if self._cached_ip_info else None,
            location=self._cached_ip_info.get('location') if self._cached_ip_info else None
        )

    def _get_user_context(self) -> UserContext:
        """
        Gather ambient user activity cues.
        
        Captures clipboard previews, recently accessed files, and active media titles.
        """
        # Clipboard
        clipboard = None
        try:
            clipboard = pyperclip.paste()
            if clipboard and len(clipboard) > 500:
                clipboard = clipboard[:500] + "..."
        except Exception:
            pass

        # Recent Files
        recent_files = []
        try:
            recent_dir = os.path.expandvars('%APPDATA%\\Microsoft\\Windows\\Recent')
            if os.path.exists(recent_dir):
                files = [os.path.join(recent_dir, f) for f in os.listdir(recent_dir)]
                # Sort by modification time
                files.sort(key=os.path.getmtime, reverse=True)
                for f in files[:5]:
                    name = os.path.splitext(os.path.basename(f))[0]
                    if name:
                        recent_files.append(name)
        except Exception:
            pass

        # Media Info (Fallback: Look for common media player window titles)
        media_info = None
        try:
            def enum_windows_callback(hwnd, titles):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        # Simple heuristics for media players
                        if any(x in title.lower() for x in ["spotify", "youtube", "vlc", "netflix"]):
                            titles.append(title)
            
            titles = []
            win32gui.EnumWindows(enum_windows_callback, titles)
            if titles:
                media_info = titles[0]
        except Exception:
            pass

        return UserContext(
            clipboard_preview=clipboard,
            recent_files=recent_files,
            media_info=media_info
        )
