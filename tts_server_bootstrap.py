import atexit
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import config

_SERVER_URL = "http://localhost:8080"
_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_server_orpheus.py")
_proc = None
_ready_evt = threading.Event()
_lock = threading.Lock()


def _ps_kill_by_port(port: int = 8080):
    """Kill any process listening on the given port (PowerShell + Get-NetTCPConnection)."""
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
                f"Select-Object -ExpandProperty OwningProcess",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return
    seen = set()
    for line in out.splitlines():
        pid = line.strip()
        if not pid or pid == "0" or pid in seen:
            continue
        seen.add(pid)
        subprocess.run(
            ["taskkill", "/F", "/PID", pid],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _ps_kill_by_script_name(script_name: str = "tts_server_orpheus.py"):
    """Kill any python.exe process whose command line contains the script name."""
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{script_name}*' }} | "
                f"Select-Object -ExpandProperty ProcessId",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return
    seen = set()
    for line in out.splitlines():
        pid = line.strip()
        if not pid or pid == "0" or pid in seen:
            continue
        seen.add(pid)
        subprocess.run(
            ["taskkill", "/F", "/PID", pid],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _kill_any_existing():
    """Aggressively terminate any old TTS server processes before we start a new one."""
    _ps_kill_by_script_name("tts_server_orpheus.py")
    _ps_kill_by_port(8080)
    time.sleep(0.5)  # let Windows reap the processes


def _poll_health(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{_SERVER_URL}/health", timeout=1.5) as r:
                if r.status == 200:
                    data = json.loads(r.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.5)
    return False


def start():
    global _proc
    with _lock:
        # If we already have a live tracked process, don't spawn another.
        if _proc is not None and _proc.poll() is None:
            return

        _kill_any_existing()
        config.custom_print("Lifespan", "TTS bootstrap launching Orpheus subprocess...")
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        _proc = subprocess.Popen(
            [sys.executable, _SERVER_SCRIPT],
            cwd=os.path.dirname(_SERVER_SCRIPT),
            creationflags=creationflags,
        )
        _ready_evt.clear()

    def _watcher():
        if _poll_health():
            config.custom_print("Lifespan", "TTS server is ready (bootstrap /health ok).")
            _ready_evt.set()
        else:
            config.custom_print("Error", "TTS server FAILED to become ready within 180s.")

    threading.Thread(target=_watcher, daemon=True, name="tts-bootstrap").start()
    atexit.register(stop)


def wait_ready(timeout=180):
    return _ready_evt.wait(timeout=timeout)


def is_ready():
    return _ready_evt.is_set()


def stop():
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            try:
                _proc.terminate()
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None
        # Also nuke any stragglers by port/script so we don't leak VRAM.
        _kill_any_existing()


def restart():
    stop()
    start()
