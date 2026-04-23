@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  WMG AIDL - Week 4 Environment Setup
::  Location: Week 4\src\utils\setup.bat
::  Run this once before opening any notebooks.
:: ============================================================

:: Resolve the root of "Week 4" — three levels up from this script's location
:: Week 4\src\utils\setup.bat  ->  up 3 = Week 4\
set "SCRIPT_DIR=%~dp0"
set "WEEK4_DIR=%SCRIPT_DIR%..\.."
set "WEEK4_DIR_ABS="
pushd "%WEEK4_DIR%" 2>nul && set "WEEK4_DIR_ABS=%CD%" && popd

if not defined WEEK4_DIR_ABS (
    echo [ERROR] Could not resolve the Week 4 directory.
    echo         Make sure this script is inside Week 4\src\utils\
    pause
    exit /b 1
)

set "VENV_DIR=%WEEK4_DIR_ABS%\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "REQUIREMENTS=%WEEK4_DIR_ABS%\requirements.txt"
set "NOTEBOOKS_DIR=%WEEK4_DIR_ABS%\src"

echo.
echo ============================================================
echo   WMG AIDL - Week 4 Environment Setup
echo ============================================================
echo.
echo   Week 4 root : %WEEK4_DIR_ABS%
echo   venv path   : %VENV_DIR%
echo   requirements: %REQUIREMENTS%
echo.

:: ── 1. Check requirements.txt exists ────────────────────────
if not exist "%REQUIREMENTS%" (
    echo [ERROR] requirements.txt not found at:
    echo         %REQUIREMENTS%
    echo         Make sure you are running this script from inside the
    echo         correct repo folder.
    pause
    exit /b 1
)

:: ── 2. Locate a working Python interpreter ──────────────────
set "SYS_PYTHON="

for %%C in (python py python3) do (
    if not defined SYS_PYTHON (
        %%C --version >nul 2>&1
        if !errorlevel! == 0 (
            set "SYS_PYTHON=%%C"
        )
    )
)

if not defined SYS_PYTHON (
    echo [ERROR] No Python interpreter found on PATH.
    echo         Please install Python 3.10 or later from https://python.org
    echo         and make sure "Add Python to PATH" is ticked during install.
    pause
    exit /b 1
)

echo [OK] System Python found: %SYS_PYTHON%
%SYS_PYTHON% --version

:: ── 3. Allow activation scripts for this session only ───────
::     Scope=Process means this never touches the machine policy.
powershell -NoProfile -NonInteractive -Command ^
    "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force" ^
    >nul 2>&1

:: ── 4. Create venv if it does not exist ─────────────────────
if exist "%VENV_PYTHON%" (
    echo [OK] Virtual environment already exists — skipping creation.
) else (
    echo [..] Creating virtual environment at %VENV_DIR% ...
    %SYS_PYTHON% -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo         Check that Python is installed correctly and you have
        echo         write access to: %WEEK4_DIR_ABS%
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

:: ── 5. Verify the venv Python is reachable ──────────────────
if not exist "%VENV_PYTHON%" (
    echo [ERROR] venv Python not found at %VENV_PYTHON%
    echo         The venv folder may be corrupt. Delete the venv folder
    echo         and run this script again.
    pause
    exit /b 1
)

:: ── 6. Upgrade pip / setuptools / wheel ─────────────────────
echo.
echo [..] Upgrading pip, setuptools, wheel ...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel --quiet
if !errorlevel! neq 0 (
    echo [WARN] Could not upgrade pip/setuptools/wheel.
    echo        Continuing anyway — this is non-fatal.
)
echo [OK] pip/setuptools/wheel up to date.

:: ── 7. Install project requirements ─────────────────────────
echo.
echo [..] Installing packages from requirements.txt ...
echo      This may take several minutes on the first run.
echo.
"%VENV_PYTHON%" -m pip install -r "%REQUIREMENTS%"
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Package installation failed.
    echo         Common causes:
    echo           - No internet connection
    echo           - A package name or version in requirements.txt is wrong
    echo           - A proxy is blocking pip
    echo         Check the output above for the specific failing package,
    echo         then ask your instructor for help.
    pause
    exit /b 1
)
echo.
echo [OK] All packages installed successfully.

:: ── 8. Register the kernel for Jupyter / VS Code ────────────
echo.
echo [..] Registering Jupyter kernel "Python (Week 4)" ...
"%VENV_PYTHON%" -m ipykernel install --user ^
    --name aidl_week4 ^
    --display-name "Python (Week 4)"
if !errorlevel! neq 0 (
    echo [WARN] ipykernel registration failed.
    echo        You may need to select the interpreter manually in VS Code.
    echo        Path to use: %VENV_PYTHON%
) else (
    echo [OK] Kernel registered as "Python (Week 4)".
)

:: ── 9. Set VS Code workspace interpreter (best-effort) ──────
::     Writes .vscode/settings.json at the Week 4 root so VS Code
::     picks up the correct interpreter automatically.
set "VSCODE_DIR=%WEEK4_DIR_ABS%\.vscode"
set "VSCODE_SETTINGS=%VSCODE_DIR%\settings.json"

if not exist "%VSCODE_DIR%" mkdir "%VSCODE_DIR%" 2>nul

:: Use forward slashes inside the JSON (Windows path with backslashes
:: must be double-escaped in JSON).
set "VENV_PYTHON_FWD=%VENV_PYTHON:\=/%"

echo { > "%VSCODE_SETTINGS%"
echo     "python.defaultInterpreterPath": "%VENV_PYTHON_FWD%", >> "%VSCODE_SETTINGS%"
echo     "python.terminal.activateEnvironment": true, >> "%VSCODE_SETTINGS%"
echo     "jupyter.notebookFileRoot": "${workspaceFolder}" >> "%VSCODE_SETTINGS%"
echo } >> "%VSCODE_SETTINGS%"

echo [OK] VS Code settings written to %VSCODE_SETTINGS%

:: ── 10. Quick sanity check ───────────────────────────────────
echo.
echo [..] Running sanity check ...
"%VENV_PYTHON%" -c "import sys; print('    Python:', sys.executable)"
if !errorlevel! neq 0 (
    echo [WARN] Sanity check failed — but setup may still be usable.
)

:: ── Done ─────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   Next steps:
echo   1. Open VS Code in the Week 4 folder (File > Open Folder)
echo   2. Open any notebook inside:
echo      %NOTEBOOKS_DIR%
echo   3. In the top-right kernel picker, choose:
echo      "Python (Week 4)"
echo.
echo   If VS Code does not show the kernel, press Ctrl+Shift+P and
echo   search for "Python: Select Interpreter", then pick:
echo   %VENV_PYTHON%
echo.
pause
exit /b 0