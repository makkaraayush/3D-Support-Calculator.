@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   3D Support Calculator - Server Shutdown Utility
echo ========================================================
echo(

set PORT=8000
set FOUND=0

REM Search for any process listening on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    set TARGET_PID=%%a
    if defined TARGET_PID (
        echo [INFO] Found server running on port %PORT% with PID !TARGET_PID!.
        echo [INFO] Terminating process !TARGET_PID!...
        taskkill /F /PID !TARGET_PID! >nul 2>&1
        set FOUND=1
    )
)

echo(
if "!FOUND!"=="1" (
    echo [SUCCESS] 3D Support Calculator server stopped successfully.
) else (
    echo [INFO] No server is currently running on port %PORT%.
)

echo(
echo Closing window...
ping -n 4 127.0.0.1 >nul
