@echo off
setlocal
cd /d "%~dp0"
title GPR Dash Application

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)

echo.
echo ============================================
echo   AVNL-OFMK GPR DASH APPLICATION
echo ============================================
echo.

%PYTHON% -c "import dash,flask,numpy,PIL,scipy,requests" >nul 2>nul
if errorlevel 1 (
    echo Required Python packages are missing.
    echo Installing requirements...
    %PYTHON% -m pip install -r requirements_dash.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting GPR Dash on http://127.0.0.1:8050
echo Press Ctrl+C to stop the server.
echo.
%PYTHON% gpr_dash_app.py
if errorlevel 1 (
    echo.
    echo ERROR: GPR Dash stopped unexpectedly.
    pause
)
endlocal
