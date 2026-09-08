@echo off
setlocal EnableExtensions
:: FFBridge counterpart to acbl-pipeline\acbl_all.bat.
:: This directory orchestrates; ingest/augment code stays in ..\elo and
:: Club-shaped outputs are mapped by ..\bridgestats-ffbridge.
::
:: Produces data files consumed by:
::   ..\bridgestats-ffbridge  (Club board-results / hand-records / lookups)
::   ..\elo                   (quality sidecars, after stage 1 copy)
::
:: Writes Club parquets to E:\bridge\data\ffbridge. bridgestats-ffbridge\u.bat
:: then copies them into that app's data\ and publishes to prod.
::
:: Set FFBRIDGE_SESSION_LIMIT for a bounded Club rebuild (smoke tests).

set "SRC=%~dp0.."
set "ELO=%SRC%\elo"
set "STATS=%SRC%\bridgestats-ffbridge"
set "ELO_PY=%ELO%\.venv\Scripts\python.exe"
set "STATS_PY=%STATS%\.venv\Scripts\python.exe"
if exist "%STATS_PY%" (
    set "CLUB_PY=%STATS_PY%"
) else (
    set "CLUB_PY=%ELO_PY%"
)
set "SOURCE=E:\bridge\data\ffbridge\data"
set "QUALITY_OUT=E:\bridge\data\ffbridge\quality_cache"
set "CLUB_OUT=E:\bridge\data\ffbridge"
set "ELO_QUALITY_DEST=%ELO%\data\ffbridge\quality_cache"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "STEP_OK=%TEMP%\ffbridge_all_step.ok"

echo ======================================================================
echo  FFBridge Full Pipeline
echo  Produces Club BridgeStats parquets from the Lancelot / quality cache.
echo ======================================================================
echo.

if not exist "%ELO_PY%" (
    echo *** FAILED: Elo venv not found: %ELO_PY%
    echo Create it with: cd ..\elo ^& python -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)
if not exist "%CLUB_PY%" (
    echo *** FAILED: Club builder Python not found: %CLUB_PY%
    exit /b 1
)
if not exist "%SOURCE%\" (
    echo *** FAILED: FFBridge cache not found: %SOURCE%
    exit /b 1
)

echo Using quality: %ELO_PY%
echo Using club:    %CLUB_PY%
echo Start: %date% %time%
echo.
call :now PIPE_T0

:: ====================================================================
:: STAGE 1: REFRESH -- discover/fetch Lancelot cache, rebuild quality
:: ====================================================================
echo [Stage 1] Refresh quality cache...
:: READS:  E:\bridge\data\ffbridge\data  (raw Lancelot / team scores)
:: WRITES: E:\bridge\data\ffbridge\quality_cache\ffbridge_quality_*.parquet
::         ..\elo\data\ffbridge\quality_cache\  (copy for the Elo app)
:: TIME:   --if-stale skips a full rebuild when today's cache is fresh.
::         Cold historical rebuild is hours (8k+ sessions). Incremental: minutes.
if not exist "%QUALITY_OUT%\" mkdir "%QUALITY_OUT%"
call :pyrun 1a "%ELO_PY%" "%ELO%\build_ffbridge_quality_parquets.py" --source-dir "%SOURCE%" --output-dir "%QUALITY_OUT%" --if-stale
if errorlevel 1 goto :error

echo   [1b] Copying quality sidecars into ..\elo\data ...
if not exist "%ELO_QUALITY_DEST%\" mkdir "%ELO_QUALITY_DEST%"
for %%F in (
    ffbridge_quality_boards.parquet
    ffbridge_quality_players.parquet
    ffbridge_quality_pairs.parquet
    ffbridge_quality_metadata.json
) do (
    if not exist "%QUALITY_OUT%\%%F" (
        echo *** FAILED: missing quality artifact %QUALITY_OUT%\%%F
        set "STEP_LABEL=1b"
        goto :error
    )
    xcopy "%QUALITY_OUT%\%%F" "%ELO_QUALITY_DEST%\" /D /Y
    if errorlevel 1 (
        set "STEP_LABEL=1b"
        goto :error
    )
)

:: ====================================================================
:: STAGE 2: CLUB BRIDGESTATS -- ACBL-shaped Stage 3 counterpart
:: ====================================================================
echo.
echo [Stage 2] Building Club BridgeStats parquets...
:: READS:  complete cached sessions via elo\ffbridge_quality_pipeline
:: WRITES: E:\bridge\data\ffbridge\ffbridge_club_board_results_augmented.parquet
::         E:\bridge\data\ffbridge\ffbridge_club_hand_records_augmented_narrow.parquet
::         E:\bridge\data\ffbridge\ffbridge_player_info.parquet
::         E:\bridge\data\ffbridge\ffbridge_clubs.parquet
:: TIME:   re-augments each complete session into slim club_session_fragments.
::         Reruns resume finished session fragments. Use FFBRIDGE_SESSION_LIMIT
::         to bound a smoke test.
set "LIMIT_ARGS="
if defined FFBRIDGE_SESSION_LIMIT set "LIMIT_ARGS=--session-limit %FFBRIDGE_SESSION_LIMIT%"
if not exist "%CLUB_OUT%\" mkdir "%CLUB_OUT%"
call :pyrun 2 "%CLUB_PY%" "%STATS%\build_ffbridge_club_parquets.py" --from-quality-cache --source-dir "%SOURCE%" --output-dir "%CLUB_OUT%" %LIMIT_ARGS%
if errorlevel 1 goto :error

:: ====================================================================
:: STAGE 3: PUBLISH -- same job as ACBL bridgestats u.bat
:: ====================================================================
echo.
echo [Stage 3] Publishing Club parquets via bridgestats-ffbridge\u.bat ...
if not exist "%STATS%\u.bat" (
    echo *** FAILED: %STATS%\u.bat not found
    set "STEP_LABEL=3"
    goto :error
)
pushd "%STATS%"
call u.bat
set "U_EXIT=%ERRORLEVEL%"
popd
if not "%U_EXIT%"=="0" (
    set "STEP_LABEL=3"
    goto :error
)

echo.
echo ======================================================================
echo  Pipeline complete: %date% %time%
call :now PIPE_T1
set /a PIPE_ELAPSED=PIPE_T1-PIPE_T0
set /a PIPE_H=PIPE_ELAPSED/3600
set /a PIPE_M=(PIPE_ELAPSED %% 3600)/60
set /a PIPE_S=PIPE_ELAPSED %% 60
echo  TIME[total]: %PIPE_ELAPSED%s (%PIPE_H%h %PIPE_M%m %PIPE_S%s)
echo.
echo  Downstream consumers and their required files:
echo.
echo  ..\bridgestats-ffbridge:
echo    ffbridge_club_board_results_augmented.parquet
echo    ffbridge_club_hand_records_augmented_narrow.parquet
echo    ffbridge_player_info.parquet
echo    ffbridge_clubs.parquet
echo.
echo  ..\elo:
echo    ffbridge_quality_boards.parquet
echo    ffbridge_quality_players.parquet
echo    ffbridge_quality_pairs.parquet
echo ======================================================================
del /q "%STEP_OK%" 2>nul
goto :eof

:: usage: call :pyrun LABEL python.exe script.py [args...]
:: Use %~1 after shift. "%1" double-quotes the already-quoted exe path,
:: and cmd then hands python.exe to another interpreter as a script.
:pyrun
set "STEP_LABEL=%~1"
shift
set "PYEXE=%~1"
del /q "%STEP_OK%" 2>nul
call :now STEP_T0
"%PYEXE%" %2 %3 %4 %5 %6 %7 %8 %9 && echo.>"%STEP_OK%"
if not exist "%STEP_OK%" exit /b 1
call :toc %STEP_LABEL%
exit /b 0

:now
for /f %%t in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "%~1=%%t"
goto :eof

:toc
call :now STEP_T1
set /a STEP_ELAPSED=STEP_T1-STEP_T0
set /a STEP_H=STEP_ELAPSED/3600
set /a STEP_M=(STEP_ELAPSED %% 3600)/60
set /a STEP_S=STEP_ELAPSED %% 60
echo   TIME[%~1]: %STEP_ELAPSED%s (%STEP_H%h %STEP_M%m %STEP_S%s) ended %date% %time%
echo.
goto :eof

:error
echo.
echo *** FAILED/INTERRUPTED at step %STEP_LABEL% %date% %time% ***
del /q "%STEP_OK%" 2>nul
exit /b 1
