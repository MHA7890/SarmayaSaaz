"""
SarmayaSaaz One-Stop Control Center & Process Launcher.

Provides a fast, zero-lag standalone GUI to:
  1. Check and install system prerequisites (Python, uv, Node.js, npm dependencies).
  2. Start & Stop both Backend (FastAPI) and Frontend (Next.js) with 1 click.
  3. Display real-time service health status and live console logs.
  4. Launch the web platform directly in your default browser.
"""
from __future__ import annotations

import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

# Root directory of the SarmayaSaaz project (supports plain python & PyInstaller .exe)
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

# Color Palette (Modern Dark Theme)
BG_DARK = "#0f172a"
BG_CARD = "#1e293b"
BG_CONSOLE = "#020617"
TEXT_PRIMARY = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
ACCENT_BLUE = "#6366f1"
ACCENT_CYAN = "#06b6d4"
ACCENT_YELLOW = "#f59e0b"
BORDER_COLOR = "#334155"


class SarmayaSaazLauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SarmayaSaaz — Control Center")
        self.root.geometry("920x680")
        self.root.minsize(800, 600)
        self.root.configure(bg=BG_DARK)

        # Process Handles
        self.backend_proc: subprocess.Popen | None = None
        self.frontend_proc: subprocess.Popen | None = None

        # Threading Log Queue
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        # State Variables
        self.backend_status = tk.StringVar(value="STOPPED")
        self.frontend_status = tk.StringVar(value="STOPPED")

        # Setup GUI Components
        self._setup_styles()
        self._build_ui()

        # Start Log Reader & Asynchronous Background Health Monitor
        self.root.after(100, self._process_log_queue)
        threading.Thread(target=self._health_monitor_worker, daemon=True).start()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Configure generic styles
        style.configure(".", background=BG_DARK, foreground=TEXT_PRIMARY, font=("Segoe UI", 10))
        style.configure("Card.TFrame", background=BG_CARD, relief="flat")
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground=TEXT_PRIMARY, background=BG_DARK)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground=TEXT_MUTED, background=BG_DARK)
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"), foreground=ACCENT_CYAN, background=BG_CARD)
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"), background=BG_CARD)

    def _build_ui(self):
        main_container = ttk.Frame(self.root, padding="16")
        main_container.pack(fill=tk.BOTH, expand=True)

        # --- Top Header ---
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 12))

        title_label = ttk.Label(header_frame, text="SarmayaSaaz Control Center", style="Header.TLabel")
        title_label.pack(side=tk.LEFT)

        version_badge = tk.Label(
            header_frame, text=" v1.0.0 ", bg=ACCENT_BLUE, fg="#ffffff",
            font=("Segoe UI", 9, "bold"), bd=0, padx=6, pady=2
        )
        version_badge.pack(side=tk.LEFT, padx=10)

        subtitle_label = ttk.Label(
            header_frame,
            text="Multi-Asset AI Forecasting Engine — Commodities, Crypto, PSX Stocks & MUFAP Funds",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        # --- Services & Controls Grid ---
        grid_frame = ttk.Frame(main_container)
        grid_frame.pack(fill=tk.X, pady=(0, 12))

        # Card 1: Service Status Card
        status_card = tk.Frame(grid_frame, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        status_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        card1_inner = ttk.Frame(status_card, style="Card.TFrame", padding="12")
        card1_inner.pack(fill=tk.BOTH, expand=True)

        ttk.Label(card1_inner, text="SERVICE STATUS", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 8))

        # Backend Status Row
        be_row = ttk.Frame(card1_inner, style="Card.TFrame")
        be_row.pack(fill=tk.X, pady=4)
        ttk.Label(be_row, text="Backend API (:8000):", font=("Segoe UI", 10), background=BG_CARD).pack(side=tk.LEFT)
        self.be_badge = tk.Label(
            be_row, textvariable=self.backend_status, bg=ACCENT_RED, fg="#ffffff",
            font=("Segoe UI", 9, "bold"), width=10, padx=4
        )
        self.be_badge.pack(side=tk.RIGHT)

        # Frontend Status Row
        fe_row = ttk.Frame(card1_inner, style="Card.TFrame")
        fe_row.pack(fill=tk.X, pady=4)
        ttk.Label(fe_row, text="Frontend Web App (:3000):", font=("Segoe UI", 10), background=BG_CARD).pack(side=tk.LEFT)
        self.fe_badge = tk.Label(
            fe_row, textvariable=self.frontend_status, bg=ACCENT_RED, fg="#ffffff",
            font=("Segoe UI", 9, "bold"), width=10, padx=4
        )
        self.fe_badge.pack(side=tk.RIGHT)

        # Links
        links_row = ttk.Frame(card1_inner, style="Card.TFrame")
        links_row.pack(fill=tk.X, pady=(8, 0))
        btn_docs = tk.Button(
            links_row, text="API Docs (/docs)", bg=BG_CARD, fg=ACCENT_CYAN,
            activebackground=BORDER_COLOR, activeforeground=TEXT_PRIMARY,
            bd=1, relief="solid", font=("Segoe UI", 9), command=lambda: webbrowser.open("http://127.0.0.1:8000/docs")
        )
        btn_docs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        # Card 2: Actions & Controls
        controls_card = tk.Frame(grid_frame, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        controls_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        card2_inner = ttk.Frame(controls_card, style="Card.TFrame", padding="12")
        card2_inner.pack(fill=tk.BOTH, expand=True)

        ttk.Label(card2_inner, text="PLATFORM CONTROLS", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 8))

        btn_box1 = ttk.Frame(card2_inner, style="Card.TFrame")
        btn_box1.pack(fill=tk.X, pady=2)

        self.btn_start = tk.Button(
            btn_box1, text="▶ Start Platform", bg=ACCENT_GREEN, fg="#ffffff",
            activebackground="#059669", activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"), bd=0, padx=12, pady=6, command=self.start_services
        )
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_stop = tk.Button(
            btn_box1, text="■ Stop Platform", bg=ACCENT_RED, fg="#ffffff",
            activebackground="#dc2626", activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"), bd=0, padx=12, pady=6, command=self.stop_services
        )
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        btn_box2 = ttk.Frame(card2_inner, style="Card.TFrame")
        btn_box2.pack(fill=tk.X, pady=(6, 0))

        self.btn_launch = tk.Button(
            btn_box2, text="🌐 Launch Website", bg=ACCENT_BLUE, fg="#ffffff",
            activebackground="#4f46e5", activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"), bd=0, padx=12, pady=6, command=self.open_website
        )
        self.btn_launch.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_deps = tk.Button(
            btn_box2, text="⚙ Install Dependencies", bg=BORDER_COLOR, fg=TEXT_PRIMARY,
            activebackground="#475569", activeforeground="#ffffff",
            font=("Segoe UI", 10), bd=0, padx=12, pady=6, command=self.install_dependencies
        )
        self.btn_deps.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # --- Console Logs Section ---
        console_frame = tk.Frame(main_container, bg=BG_CARD, bd=1, relief="solid", highlightbackground=BORDER_COLOR)
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        log_header = ttk.Frame(console_frame, style="Card.TFrame", padding=(12, 8, 12, 4))
        log_header.pack(fill=tk.X)

        ttk.Label(log_header, text="LIVE SYSTEM LOGS", style="Section.TLabel").pack(side=tk.LEFT)

        btn_clear = tk.Button(
            log_header, text="Clear Log", bg=BG_CARD, fg=TEXT_MUTED,
            activebackground=BORDER_COLOR, activeforeground=TEXT_PRIMARY,
            bd=0, font=("Segoe UI", 8), command=self._clear_logs
        )
        btn_clear.pack(side=tk.RIGHT)

        # Log Text Box with Scrollbar
        log_box_frame = ttk.Frame(console_frame, style="Card.TFrame", padding=(12, 0, 12, 12))
        log_box_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_box_frame, bg=BG_CONSOLE, fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY, font=("Consolas", 10),
            bd=0, highlightthickness=0, wrap=tk.WORD
        )
        scrollbar = ttk.Scrollbar(log_box_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Log formatting tags
        self.log_text.tag_config("BACKEND", foreground=ACCENT_CYAN)
        self.log_text.tag_config("FRONTEND", foreground=ACCENT_YELLOW)
        self.log_text.tag_config("SYSTEM", foreground=ACCENT_GREEN)
        self.log_text.tag_config("ERROR", foreground=ACCENT_RED)

        # Initial Welcome Message
        self.log("SYSTEM", "==========================================================================")
        self.log("SYSTEM", " SarmayaSaaz Control Center Ready")
        self.log("SYSTEM", f" Project Root: {PROJECT_ROOT}")
        self.log("SYSTEM", " Click 'Start Platform' to launch backend and frontend.")
        self.log("SYSTEM", "==========================================================================")

        # Check initial dependencies fast
        backend_ok, frontend_ok, missing = self._check_dependencies_fast()
        if not (backend_ok and frontend_ok):
            self.log("SYSTEM", "⚠️ Prerequisites check: missing dependencies detected.")
            for item in missing:
                self.log("SYSTEM", f"   - {item}")
            self.log("SYSTEM", "👉 Click '⚙ Install Dependencies' to set up required components.")

    # --- Logging Helpers ---
    def log(self, category: str, message: str):
        self.log_queue.put((category, message))

    def _process_log_queue(self):
        while not self.log_queue.empty():
            category, msg = self.log_queue.get_nowait()
            tag = category if category in ("BACKEND", "FRONTEND", "SYSTEM", "ERROR") else "SYSTEM"
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] [{category:<8}] {msg}\n", tag)
            self.log_text.see(tk.END)
        self.root.after(100, self._process_log_queue)

    def _clear_logs(self):
        self.log_text.delete("1.0", tk.END)

    # --- Fast Non-Blocking Background Health Worker ---
    def _health_monitor_worker(self):
        while True:
            # Check Port 8000
            be_open = self._check_port_open("127.0.0.1", 8000)
            # Check Port 3000
            fe_open = self._check_port_open("127.0.0.1", 3000) or self._check_port_open("localhost", 3000)

            # Schedule UI badge updates safely on main thread
            self.root.after(0, self._update_status_badges, be_open, fe_open)
            time.sleep(1.5)

    def _check_port_open(self, host: str, port: int, timeout: float = 0.3) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def _update_status_badges(self, be_open: bool, fe_open: bool):
        if be_open:
            self.backend_status.set("ONLINE")
            self.be_badge.configure(bg=ACCENT_GREEN)
        elif self.backend_proc is not None and self.backend_proc.poll() is None:
            self.backend_status.set("STARTING")
            self.be_badge.configure(bg=ACCENT_YELLOW)
        else:
            self.backend_status.set("STOPPED")
            self.be_badge.configure(bg=ACCENT_RED)

        if fe_open:
            self.frontend_status.set("ONLINE")
            self.fe_badge.configure(bg=ACCENT_GREEN)
        elif self.frontend_proc is not None and self.frontend_proc.poll() is None:
            self.frontend_status.set("STARTING")
            self.fe_badge.configure(bg=ACCENT_YELLOW)
        else:
            self.frontend_status.set("STOPPED")
            self.fe_badge.configure(bg=ACCENT_RED)

    # --- Robust Python Resolver for Backend Execution ---
    def _get_backend_python_command(self) -> list[str] | None:
        # 1. Look for existing .venv python in PROJECT_ROOT with uvicorn installed
        venv_py = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        uvicorn_exe = PROJECT_ROOT / ".venv" / ("Scripts/uvicorn.exe" if os.name == "nt" else "bin/uvicorn")
        uvicorn_pkg = PROJECT_ROOT / ".venv/Lib/site-packages/uvicorn"
        if venv_py.exists() and (uvicorn_exe.exists() or uvicorn_pkg.exists()):
            return [str(venv_py), "-m", "uvicorn"]

        # 2. Look for uv executable in system PATH
        uv_bin = shutil.which("uv")
        if uv_bin:
            return [uv_bin, "run", "uvicorn"]

        # 3. Look for system python executables in system PATH
        for py_name in ["python", "python3", "py"]:
            py_path = shutil.which(py_name)
            if py_path:
                return [py_path, "-m", "uvicorn"]

        # 4. Fallback to sys.executable ONLY if running as source script (not frozen .exe)
        if not getattr(sys, "frozen", False):
            return [sys.executable, "-m", "uvicorn"]

        return None

    # --- Instant Dependency Pre-Check & Installer ---
    def _check_dependencies_fast(self) -> tuple[bool, bool, list[str]]:
        missing = []

        # 1. Fast Check Python .venv & uvicorn
        venv_dir = PROJECT_ROOT / ".venv"
        python_bin = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        uvicorn_exe = venv_dir / ("Scripts/uvicorn.exe" if os.name == "nt" else "bin/uvicorn")
        uvicorn_pkg = venv_dir / "Lib/site-packages/uvicorn"
        backend_ok = venv_dir.exists() and python_bin.exists() and (uvicorn_exe.exists() or uvicorn_pkg.exists())

        if not backend_ok:
            if not venv_dir.exists():
                missing.append("Python virtual environment (.venv) is missing.")
            else:
                missing.append("Python dependencies (uvicorn, fastapi, etc.) are missing in .venv.")

        # 2. Fast Check Frontend node_modules
        node_modules = PROJECT_ROOT / "frontend" / "node_modules"
        frontend_ok = node_modules.exists() and (node_modules / "next").exists()

        if not frontend_ok:
            missing.append("Frontend node_modules directory is missing.")

        return backend_ok, frontend_ok, missing

    def install_dependencies(self):
        def _run_installer():
            self.log("SYSTEM", "Verifying installed dependencies...")
            backend_ok, frontend_ok, missing = self._check_dependencies_fast()

            if backend_ok and frontend_ok:
                self.log("SYSTEM", "✅ Python backend virtual environment (.venv) is ready.")
                self.log("SYSTEM", "✅ Frontend node_modules is ready.")
                self.log("SYSTEM", "==========================================================================")
                self.log("SYSTEM", "🎉 All dependencies are already installed! No download required.")
                self.log("SYSTEM", " Click 'Start Platform' to launch SarmayaSaaz.")
                self.log("SYSTEM", "==========================================================================")
                messagebox.showinfo(
                    "Dependencies Up to Date",
                    "All required Python and Node.js dependencies are already installed!\n\nClick 'Start Platform' to launch SarmayaSaaz."
                )
                return

            for msg in missing:
                self.log("SYSTEM", f"⚠️ {msg}")

            # Check Python & uv if backend missing
            if not backend_ok:
                uv_bin = shutil.which("uv")
                if uv_bin:
                    self.log("SYSTEM", "Running 'uv sync --extra dev' for Python backend...")
                    try:
                        p = subprocess.Popen(
                            [uv_bin, "sync", "--extra", "dev"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=str(PROJECT_ROOT)
                        )
                        if p.stdout:
                            for line in iter(p.stdout.readline, ""):
                                if line: self.log("BACKEND", line.strip())
                        p.wait()
                        self.log("SYSTEM", "Python backend dependencies installed successfully via uv.")
                    except Exception as e:
                        self.log("ERROR", f"uv sync failed: {e}")
                else:
                    sys_py = None
                    for py_name in ["python", "python3", "py"]:
                        p_path = shutil.which(py_name)
                        if p_path:
                            sys_py = p_path
                            break

                    if not sys_py and not getattr(sys, "frozen", False):
                        sys_py = sys.executable

                    if sys_py:
                        self.log("SYSTEM", f"Found system Python: {sys_py}. Creating/verifying virtual environment...")
                        venv_dir = PROJECT_ROOT / ".venv"
                        try:
                            p = subprocess.Popen(
                                [sys_py, "-m", "venv", str(venv_dir)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(PROJECT_ROOT)
                            )
                            p.wait()

                            venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                            if venv_py.exists():
                                self.log("SYSTEM", "Upgrading pip and setuptools in .venv...")
                                subprocess.run(
                                    [str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                                    capture_output=True
                                )

                                self.log("SYSTEM", "Installing Python dependencies into .venv via pip...")
                                pip_cmd = [str(venv_py), "-m", "pip", "install", "-e", "."]
                                p2 = subprocess.Popen(
                                    pip_cmd,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, cwd=str(PROJECT_ROOT)
                                )
                                if p2.stdout:
                                    for line in iter(p2.stdout.readline, ""):
                                        if line: self.log("BACKEND", line.strip())
                                rc = p2.wait()

                                if rc != 0:
                                    self.log("SYSTEM", "⚠️ Editable package build failed. Falling back to direct dependency installation...")
                                    fallback_pkgs = [
                                        "fastapi", "uvicorn[standard]", "pydantic", "pydantic-settings",
                                        "httpx", "python-dotenv", "scikit-learn==1.7.2", "numpy>=2.0,<2.4",
                                        "pandas>=2.2,<3.0", "torch>=2.4,<3.0", "xgboost==3.4.0", "lightgbm",
                                        "catboost", "joblib", "ta", "yfinance", "websocket-client", "openpyxl", "shap"
                                    ]
                                    p3 = subprocess.Popen(
                                        [str(venv_py), "-m", "pip", "install"] + fallback_pkgs,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, cwd=str(PROJECT_ROOT)
                                    )
                                    if p3.stdout:
                                        for line in iter(p3.stdout.readline, ""):
                                            if line: self.log("BACKEND", line.strip())
                                    rc3 = p3.wait()
                                    if rc3 == 0:
                                        self.log("SYSTEM", "Python backend dependencies installed successfully via direct pip fallback.")
                                    else:
                                        self.log("ERROR", "Direct pip fallback failed. Check logs above.")
                                else:
                                    self.log("SYSTEM", "Python backend dependencies installed successfully via pip.")
                            else:
                                self.log("ERROR", "Failed to locate python inside created .venv directory.")
                        except Exception as e:
                            self.log("ERROR", f"Virtualenv / pip installation failed: {e}")
                    else:
                        self.log("ERROR", "Python is not installed or not in PATH! Please install Python 3.12 (https://www.python.org).")

            # Check Node.js / npm if frontend missing
            if not frontend_ok:
                npm_bin = shutil.which("npm.cmd") or shutil.which("npm")
                if not npm_bin:
                    self.log("ERROR", "npm not found in PATH! Please install Node.js (https://nodejs.org).")
                else:
                    frontend_dir = PROJECT_ROOT / "frontend"
                    self.log("SYSTEM", "Running 'npm install' for Next.js frontend...")
                    try:
                        p = subprocess.Popen(
                            [npm_bin, "install"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=str(frontend_dir), shell=os.name == "nt"
                        )
                        if p.stdout:
                            for line in iter(p.stdout.readline, ""):
                                if line: self.log("FRONTEND", line.strip())
                        p.wait()
                        self.log("SYSTEM", "Frontend dependencies installed successfully.")
                    except Exception as e:
                        self.log("ERROR", f"npm install failed: {e}")

            self.log("SYSTEM", "Setup complete! Click 'Start Platform' to run SarmayaSaaz.")

        threading.Thread(target=_run_installer, daemon=True).start()

    # --- Start Services ---
    def start_services(self):
        # 1. Start Backend API
        if self.backend_proc and self.backend_proc.poll() is None:
            self.log("SYSTEM", "Backend process is already running.")
        else:
            self.log("SYSTEM", "Resolving Python environment for Backend API...")
            cmd_prefix = self._get_backend_python_command()
            if not cmd_prefix:
                self.log("ERROR", "Cannot start backend: Python virtual environment (.venv) or Python executable not found.")
                self.log("ERROR", "Please click '⚙ Install Dependencies' to set up the environment, or install Python 3.12.")
                self.backend_status.set("STOPPED")
                self.be_badge.configure(bg=ACCENT_RED)
            else:
                cmd = cmd_prefix + ["backend.main:app", "--port", "8000"]
                self.log("SYSTEM", f"Launching Backend API ({' '.join(cmd)})...")
                try:
                    self.backend_proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, cwd=str(PROJECT_ROOT), bufsize=1
                    )
                    self.backend_status.set("STARTING")
                    self.be_badge.configure(bg=ACCENT_YELLOW)

                    def _stream_backend_logs(proc: subprocess.Popen):
                        if proc.stdout:
                            for line in iter(proc.stdout.readline, ""):
                                if line: self.log("BACKEND", line.strip())

                    threading.Thread(target=_stream_backend_logs, args=(self.backend_proc,), daemon=True).start()
                except Exception as e:
                    self.log("ERROR", f"Failed to start backend process: {e}")
                    self.backend_status.set("STOPPED")
                    self.be_badge.configure(bg=ACCENT_RED)

        # 2. Start Frontend Web App
        if self.frontend_proc and self.frontend_proc.poll() is None:
            self.log("SYSTEM", "Frontend process is already running.")
        else:
            self.log("SYSTEM", "Launching Frontend Web App (npm run dev)...")
            npm_bin = shutil.which("npm.cmd") or shutil.which("npm")
            frontend_dir = PROJECT_ROOT / "frontend"

            if not npm_bin:
                self.log("ERROR", "Cannot start frontend: npm is not installed or not found in system PATH.")
                self.log("ERROR", "Please install Node.js (https://nodejs.org) to run the frontend.")
                self.frontend_status.set("STOPPED")
                self.fe_badge.configure(bg=ACCENT_RED)
                return

            try:
                self.frontend_proc = subprocess.Popen(
                    [npm_bin, "run", "dev"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(frontend_dir), shell=os.name == "nt", bufsize=1
                )
                self.frontend_status.set("STARTING")
                self.fe_badge.configure(bg=ACCENT_YELLOW)

                def _stream_frontend_logs(proc: subprocess.Popen):
                    if proc.stdout:
                        for line in iter(proc.stdout.readline, ""):
                            if line: self.log("FRONTEND", line.strip())

                threading.Thread(target=_stream_frontend_logs, args=(self.frontend_proc,), daemon=True).start()
            except Exception as e:
                self.log("ERROR", f"Failed to start frontend process: {e}")
                self.frontend_status.set("STOPPED")
                self.fe_badge.configure(bg=ACCENT_RED)

    # --- Stop Services ---
    def stop_services(self):
        self.log("SYSTEM", "Stopping all platform services...")

        def _kill_proc(proc: subprocess.Popen | None, name: str):
            if proc and proc.poll() is None:
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                    else:
                        subprocess.run(["kill", "-9", str(proc.pid)], capture_output=True)
                except Exception as e:
                    self.log("ERROR", f"Error stopping {name}: {e}")

        if self.backend_proc:
            _kill_proc(self.backend_proc, "Backend")
            self.backend_proc = None
            self.log("SYSTEM", "Backend process stopped.")

        if self.frontend_proc:
            _kill_proc(self.frontend_proc, "Frontend")
            self.frontend_proc = None
            self.log("SYSTEM", "Frontend process stopped.")

        self.backend_status.set("STOPPED")
        self.frontend_status.set("STOPPED")
        self.be_badge.configure(bg=ACCENT_RED)
        self.fe_badge.configure(bg=ACCENT_RED)

    # --- Open Website ---
    def open_website(self):
        url = "http://localhost:3000"
        self.log("SYSTEM", f"Opening web application in browser ({url})...")
        webbrowser.open(url)

    def _on_closing(self):
        if (self.backend_proc and self.backend_proc.poll() is None) or (self.frontend_proc and self.frontend_proc.poll() is None):
            if messagebox.askokcancel("Quit SarmayaSaaz Control Center", "Stopping SarmayaSaaz launcher will shut down running backend and frontend services. Continue?"):
                self.stop_services()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = SarmayaSaazLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
