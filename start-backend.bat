@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Creating one...
  python -m venv .venv
  if errorlevel 1 goto :fail
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
)

echo Starting backend on http://localhost:8000 ...
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
goto :eof

:fail
echo Failed to create the virtual environment.
pause