@echo off
title InSpecAI Launcher
color 0A
echo.
echo  ============================================
echo     InSpecAI - Starting All Services
echo  ============================================
echo.

:: Get the directory where this .bat lives
set ROOT=%~dp0

:: -----------------------------------------------
:: 1. Image Backend  (port 8000)
:: -----------------------------------------------
echo [1/3] Starting Image Backend on port 8000...
start "InSpecAI - Image Backend :8000" cmd /k "cd /d "%ROOT%backend" && (if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat) && uvicorn server:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul

:: -----------------------------------------------
:: 2. Video Backend  (port 8001)
:: -----------------------------------------------
echo [2/3] Starting Video Backend on port 8001...
start "InSpecAI - Video Backend :8001" cmd /k "cd /d "%ROOT%backend_vid" && (if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat) && uvicorn api:app --host 127.0.0.1 --port 8001 --reload"

timeout /t 2 /nobreak >nul

:: -----------------------------------------------
:: 3. Next.js Frontend  (port 3000)
:: -----------------------------------------------
echo [3/3] Starting Next.js Frontend on port 3000...
start "InSpecAI - Frontend :3000" cmd /k "cd /d "%ROOT%inspecai" && npm run dev"

echo.
echo  ============================================
echo   All services launched in separate windows.
echo   Image API  : http://localhost:8000
echo   Video API  : http://localhost:8001
echo   Frontend   : http://localhost:3000
echo  ============================================
echo.
pause
