@echo off
setlocal
cd /d "%~dp0\frontend"

if not exist "node_modules" (
  echo Installing frontend dependencies...
  call npm install
)

if not exist ".env.local" (
  echo NEXT_PUBLIC_API_URL=http://localhost:8000> .env.local
  echo NEXT_PUBLIC_USE_MOCK_API=false>> .env.local
)

echo Starting frontend on http://localhost:3000 ...
call npm run dev