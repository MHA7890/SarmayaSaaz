"""
Build script to compile SarmayaSaaz_Launcher into a standalone executable.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build():
    print("==========================================================")
    print(" Building SarmayaSaaz Control Center Executable")
    print("==========================================================")

    # Ensure pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        uv_bin = shutil.which("uv")
        if uv_bin:
            subprocess.check_call([uv_bin, "pip", "install", "pyinstaller"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    pyw_path = PROJECT_ROOT / "SarmayaSaaz_Launcher.pyw"
    if not pyw_path.exists():
        print(f"Error: {pyw_path} not found!")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconsole",
        "--onefile",
        "--name", "SarmayaSaaz_Launcher",
        str(pyw_path)
    ]

    print(f"Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

    dist_exe = PROJECT_ROOT / "dist" / ("SarmayaSaaz_Launcher.exe" if os.name == "nt" else "SarmayaSaaz_Launcher")
    target_exe = PROJECT_ROOT / ("SarmayaSaaz_Launcher.exe" if os.name == "nt" else "SarmayaSaaz_Launcher")

    if dist_exe.exists():
        try:
            shutil.copy(dist_exe, target_exe)
            print(f"\nSuccess! Compiled launcher saved to:\n  {target_exe}")
        except PermissionError:
            print(f"\nWarning: Could not overwrite {target_exe} because it is currently running.")
            print(f"The freshly compiled executable is available at:\n  {dist_exe}")

if __name__ == "__main__":
    build()
