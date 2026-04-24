"""
WMG AIDL - Week 4 Environment Setup
Location: Week 4/src/utils/setup.py

Run once before opening any notebooks:
    python setup.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path


# ── ANSI colours (disabled automatically on Windows if not supported) ─────────
def _supports_colour():
    return (
        sys.platform != "win32" or "ANSICON" in os.environ or "WT_SESSION" in os.environ
    )


RESET = "\033[0m" if _supports_colour() else ""
GREEN = "\033[92m" if _supports_colour() else ""
YELLOW = "\033[93m" if _supports_colour() else ""
RED = "\033[91m" if _supports_colour() else ""
BOLD = "\033[1m" if _supports_colour() else ""


def ok(msg):
    print(f"{GREEN}[OK]{RESET}   {msg}")


def warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def info(msg):
    print(f"[..]   {msg}")


def error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")


def fatal(msg):
    print()
    error(msg)
    print()
    input("Press Enter to exit...")
    sys.exit(1)


# ── Path resolution ───────────────────────────────────────────────────────────
# This file lives at:  Week 4/src/utils/setup.py
# Week 4 root is:      three levels up
SCRIPT_DIR = Path(__file__).resolve().parent  # Week 4/src/utils
WEEK4_DIR = SCRIPT_DIR.parent.parent  # Week 4/
VENV_DIR = WEEK4_DIR / "venv_rag"
REQUIREMENTS = WEEK4_DIR / "requirements.txt"
NOTEBOOKS_DIR = WEEK4_DIR / "src"
VSCODE_DIR = WEEK4_DIR / ".vscode"
VSCODE_SETTINGS = VSCODE_DIR / "settings.json"

if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"


def run(cmd, check=True, capture=False):
    """Run a subprocess command. Returns CompletedProcess."""
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


# ─────────────────────────────────────────────────────────────────────────────


def print_header():
    print()
    print(f"{BOLD}============================================================{RESET}")
    print(f"{BOLD}  WMG AIDL - Week 4 Environment Setup{RESET}")
    print(f"{BOLD}============================================================{RESET}")
    print()
    print(f"  Week 4 root  : {WEEK4_DIR}")
    print(f"  venv path    : {VENV_DIR}")
    print(f"  requirements : {REQUIREMENTS}")
    print()


# ── Step 1: Check requirements.txt ───────────────────────────────────────────
def check_requirements():
    info("Checking requirements.txt ...")
    if not REQUIREMENTS.exists():
        fatal(
            f"requirements.txt not found at:\n"
            f"        {REQUIREMENTS}\n"
            f"        Make sure you are running this script from inside the\n"
            f"        correct repo folder."
        )
    ok("requirements.txt found.")


# ── Step 2: Check system Python ──────────────────────────────────────────────
def check_system_python():
    """Confirm we are running a sane Python version."""
    info("Checking Python version ...")
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        fatal(
            f"Python 3.10 or later is required.\n"
            f"        You are running Python {major}.{minor}.\n"
            f"        Please install a newer version from https://python.org"
        )
    ok(f"Python {major}.{minor} — OK ({sys.executable})")


# ── Step 3: Create venv if absent ────────────────────────────────────────────
def create_venv():
    if VENV_PYTHON.exists():
        ok("Virtual environment already exists — skipping creation.")
        return

    info(f"Creating virtual environment at {VENV_DIR} ...")
    try:
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError:
        fatal(
            "Failed to create virtual environment.\n"
            "        Check that you have write access to:\n"
            f"        {WEEK4_DIR}"
        )

    if not VENV_PYTHON.exists():
        fatal(
            "venv was created but the Python binary is missing.\n"
            "        Try deleting the venv folder and running this script again."
        )

    ok("Virtual environment created.")


# ── Step 4: Upgrade pip / setuptools / wheel ─────────────────────────────────
def upgrade_pip():
    info("Upgrading pip, setuptools, wheel ...")
    try:
        run(
            [
                str(VENV_PYTHON),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
                "--quiet",
            ]
        )
        ok("pip/setuptools/wheel up to date.")
    except subprocess.CalledProcessError:
        warn("Could not upgrade pip/setuptools/wheel — continuing anyway.")


# ── Step 5: Install requirements ─────────────────────────────────────────────
def install_requirements():
    print()
    info("Installing packages from requirements.txt ...")
    info("This may take several minutes on the first run.")
    print()
    try:
        run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    except subprocess.CalledProcessError:
        fatal(
            "Package installation failed.\n"
            "        Common causes:\n"
            "          - No internet connection\n"
            "          - A package name or version in requirements.txt is wrong\n"
            "          - A proxy is blocking pip\n"
            "        Check the output above for the specific failing package,\n"
            "        then ask your instructor for help."
        )
    print()
    ok("All packages installed successfully.")


# ── Step 6: Register Jupyter kernel ──────────────────────────────────────────
def register_kernel():
    print()
    info('Registering Jupyter kernel "Python (Week 4)" ...')
    try:
        run(
            [
                str(VENV_PYTHON),
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                "aidl_week4",
                "--display-name",
                "Python (Week 4)",
            ]
        )
        ok('Kernel registered as "Python (Week 4)".')
    except subprocess.CalledProcessError:
        warn(
            "ipykernel registration failed.\n"
            "        You may need to select the interpreter manually in VS Code.\n"
            f"        Path to use: {VENV_PYTHON}"
        )


# ── Step 7: Write .vscode/settings.json ──────────────────────────────────────
def write_vscode_settings():
    print()
    info("Writing VS Code settings ...")
    try:
        VSCODE_DIR.mkdir(exist_ok=True)
        settings = {
            "python.defaultInterpreterPath": str(VENV_PYTHON),
            "python.terminal.activateEnvironment": True,
            "jupyter.notebookFileRoot": "${workspaceFolder}",
        }
        with open(VSCODE_SETTINGS, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        ok(f"VS Code settings written to {VSCODE_SETTINGS}")
    except OSError as e:
        warn(
            f"Could not write VS Code settings: {e}\n"
            "        You can set the interpreter manually in VS Code."
        )


# ── Step 8: Sanity check ──────────────────────────────────────────────────────
def sanity_check():
    print()
    info("Running sanity check ...")
    try:
        result = run(
            [str(VENV_PYTHON), "-c", "import sys; print(sys.executable)"],
            capture=True,
        )
        ok(f"Python : {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        warn("Sanity check failed — but setup may still be usable.")


# ── Done ──────────────────────────────────────────────────────────────────────
def print_footer():
    print()
    print(f"{BOLD}============================================================{RESET}")
    print(f"{GREEN}{BOLD}  Setup complete!{RESET}")
    print(f"{BOLD}============================================================{RESET}")
    print()
    print("  Next steps:")
    print("  1. Open VS Code in the Week 4 folder (File > Open Folder)")
    print(f"  2. Open any notebook inside:")
    print(f"     {NOTEBOOKS_DIR}")
    print("  3. In the top-right kernel picker, choose:")
    print('     "Python (Week 4)"')
    print()
    print("  If VS Code does not show the kernel, press Ctrl+Shift+P and")
    print('  search for "Python: Select Interpreter", then pick:')
    print(f"  {VENV_PYTHON}")
    print()
    input("Press Enter to exit...")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_header()
    check_requirements()
    check_system_python()
    create_venv()
    upgrade_pip()
    install_requirements()
    register_kernel()
    write_vscode_settings()
    sanity_check()
    print_footer()
