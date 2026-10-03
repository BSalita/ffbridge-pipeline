@echo off
setlocal EnableExtensions
:: Bulk-update recent FFBridge games into e:\bridge\data\ffbridge\recent.
:: Schedule one of: hour, day, week, quarter.
cd /d "%~dp0..\elo"
set "EVERY=%~1"
if "%EVERY%"=="" set "EVERY=day"
if not exist ".venv\Scripts\python.exe" (
    echo *** FAILED: Elo venv not found
    exit /b 1
)
".venv\Scripts\python.exe" ffbridge_recent_update.py --every %EVERY%
exit /b %ERRORLEVEL%
