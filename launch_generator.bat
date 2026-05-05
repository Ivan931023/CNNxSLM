@echo off
REM Windows launcher for the Flattop Beam Generator (Zernike aberrations).
REM Double-click this file to start the server and open the browser.

cd /d "%~dp0"

set PORT=8766
set URL=http://localhost:%PORT%/Flattop_beam_with_Zernike_aberrations.html

REM Kill any process already listening on the port (best-effort).
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%P >nul 2>&1
)

REM Prefer the project venv if it exists, otherwise system python.
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo Starting Flattop Beam Generator server...
echo Loading tensors.mat -- this takes several seconds (longer on CPU-only).
echo.

REM Start the server in a new window so closing it stops the server.
start "Flattop Generator Server" cmd /k "%PY% app\generator_server.py"

REM Poll the URL until the server is up (max ~30s).
set /a tries=0
:wait_loop
timeout /t 1 /nobreak >nul
set /a tries+=1
curl -s -o nul --max-time 1 "%URL%" >nul 2>&1
if %errorlevel%==0 goto ready
if %tries% LSS 30 goto wait_loop

:ready
echo.
echo Generator running at: %URL%
echo.
start "" "%URL%"

echo Close the "Flattop Generator Server" window to stop the server.
echo.
pause
