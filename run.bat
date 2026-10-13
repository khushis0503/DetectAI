@echo off
REM DetectAI - double-click to start the server and open the website.
title DetectAI server
cd /d "%~dp0"

REM First run on a new computer: create the virtual environment and install packages
if not exist "venv\Scripts\python.exe" (
    echo Setting up DetectAI for the first time - this takes a few minutes...
    python -m venv venv
    venv\Scripts\python -m pip install --upgrade pip
    venv\Scripts\python -m pip install -r backend\requirements.txt
)

echo.
echo  DetectAI is starting at http://127.0.0.1:8000
echo  Keep this window open during the demo. Close it to stop the server.
echo.

REM Open the browser a few seconds later, once the models have loaded
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://127.0.0.1:8000"

venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
pause
