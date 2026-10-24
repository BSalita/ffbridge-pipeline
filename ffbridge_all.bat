@echo off
setlocal EnableExtensions
:: FFBridge counterpart to acbl-pipeline\acbl_all.bat.
:: This directory orchestrates; ingest/augment code stays in ..\elo and
:: Club-shaped outputs are mapped by ..\bridgestats-ffbridge.
::
:: Produces data files consumed by:
::   ..\bridgestats-ffbridge  (Club board-results / hand-records / lookups)
::   ..\elo                   (quality sidecars, after stage 1 copy)
::   E:\bridge\data\ffbridge\postmortem_archive_*  (stage 4 postmortem sessions)
::   E:\bridge\data\ffbridge\SavedModels  (stage 5 checkpoints and history tables)
::
:: Writes Club parquets to E:\bridge\data\ffbridge. bridgestats-ffbridge\u.bat
:: then copies them into that app's data\ and publishes to prod.
::
:: Set FFBRIDGE_SESSION_LIMIT for a bounded Club rebuild (smoke tests).

set "SRC=%~dp0.."
set "ELO=%SRC%\elo"
set "STATS=%SRC%\bridgestats-ffbridge"
set "ACBL=%SRC%\acbl-pipeline"
set "ELO_PY=%ELO%\.venv\Scripts\python.exe"
set "STATS_PY=%STATS%\.venv\Scripts\python.exe"
set "ACBL_PY=%ACBL%\.venv\Scripts\python.exe"
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
::         E:\bridge\data\ffbridge\club_session_fragments\  (Club fragments for stage 2)
::         E:\bridge\data\ffbridge\ffbridge_cache_audit.json  (complete sessions)
::         E:\bridge\data\ffbridge\ffbridge_identity_map.json  (Lancelot-to-Elo aliases)
::         ..\elo\data\ffbridge\quality_cache\  (copy for the Elo app)
:: TIME:   --if-stale skips a full rebuild when today's cache is fresh.
::         The audit reuses ffbridge_cache_audit.json and runs again only when
::         the fetch wrote files. Sessions the last build found unusable are
::         not reloaded (--retry-unsupported retries them). New sessions go in
::         chunks of 128: every new deal is solved once (batched ddss DD, Par,
::         SD split across workers), the hand-records cache is saved once per
::         chunk, then FFBRIDGE_QUALITY_WORKERS processes (default 16) augment
::         the sessions and write quality and Club fragments in one pass.
::         Cold historical rebuild is hours (8k+ sessions). Incremental: minutes.
set "QUALITY_WORKERS=16"
if defined FFBRIDGE_QUALITY_WORKERS set "QUALITY_WORKERS=%FFBRIDGE_QUALITY_WORKERS%"
if not exist "%QUALITY_OUT%\" mkdir "%QUALITY_OUT%"
call :pyrun 1a "%ELO_PY%" "%ELO%\build_ffbridge_quality_parquets.py" --source-dir "%SOURCE%" --output-dir "%QUALITY_OUT%" --club-output-dir "%CLUB_OUT%" --workers %QUALITY_WORKERS% --if-stale
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
:: TIME:   stage 1 already wrote club_session_fragments for every session it
::         augmented, so this step mostly concatenates fragments. It augments
::         only sessions with no current fragment (e.g. ones in the training
::         parquet). --if-stale skips the rewrite when the candidate sessions
::         and their fragments are unchanged since the last build
::         (ffbridge_club_manifest.json). Use FFBRIDGE_SESSION_LIMIT to bound
::         a smoke test. FFBRIDGE_CLUB_WORKERS defaults to 8.
set "LIMIT_ARGS="
if defined FFBRIDGE_SESSION_LIMIT set "LIMIT_ARGS=--session-limit %FFBRIDGE_SESSION_LIMIT%"
set "WORKER_ARGS=--workers 8"
if defined FFBRIDGE_CLUB_WORKERS set "WORKER_ARGS=--workers %FFBRIDGE_CLUB_WORKERS%"
if not exist "%CLUB_OUT%\" mkdir "%CLUB_OUT%"
call :pyrun 2 "%CLUB_PY%" "%STATS%\build_ffbridge_club_parquets.py" --from-quality-cache --source-dir "%SOURCE%" --output-dir "%CLUB_OUT%" %LIMIT_ARGS% %WORKER_ARGS% --if-stale
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

:: ====================================================================
:: STAGE 4: POSTMORTEM ARCHIVE -- sessions the FFBridge postmortem serves
:: ====================================================================
echo.
echo [Stage 4] Building the postmortem archive...
:: READS:  E:\bridge\data\ffbridge\data  (raw Lancelot ranking/team JSON)
::         E:\bridge\data\ffbridge\ffbridge_cache_audit.json from stage 1, so
::         sessions known complete are not re-audited.
:: WRITES: E:\bridge\data\ffbridge\postmortem_archive_flat\
::         E:\bridge\data\ffbridge\postmortem_archive_hierarchical\
::           (fragments and manifest.parquet; metadata.json is its own
::           layout seed, so the column layout never changes here)
:: TIME:   augments only sessions on or after FFBRIDGE_ARCHIVE_SINCE
::         (default 2026-01-01) missing from either archive.
::         MEASURED 2026-10-24: 8-9.5s augment per session (was 200-400s,
::         almost all in identify_best_contracts_by_ev). Deals are solved
::         fresh, not read from stage 1's hand-records cache: that cache's
::         ParNumber counts contracts differently.
::         FFBRIDGE_ARCHIVE_WORKERS defaults to 8; each worker gets
::         cores/workers ddss threads (DDSS_MAX_THREADS).
::         Compaction is skipped: u.bat publishes only the manifest and
::         fragments, and compaction needs tens of GB of RAM.
:: PUBLISH: ..\elo\u.bat copies the hierarchical archive to src\data and X1.
::         It keeps sessions the live service added since the last build.
set "PM=%SRC%\postmortem-ffbridge"
set "HIER=%CLUB_OUT%\postmortem_archive_hierarchical"
set "ARCHIVE_WORKERS=8"
if defined FFBRIDGE_ARCHIVE_WORKERS set "ARCHIVE_WORKERS=%FFBRIDGE_ARCHIVE_WORKERS%"
set "ARCHIVE_SINCE=2026-01-01"
if defined FFBRIDGE_ARCHIVE_SINCE set "ARCHIVE_SINCE=%FFBRIDGE_ARCHIVE_SINCE%"
if not exist "%HIER%\metadata.json" (
    echo *** FAILED: hierarchical archive layout not found: %HIER%\metadata.json
    set "STEP_LABEL=4"
    goto :error
)
set "PYTHONPATH_SAVED=%PYTHONPATH%"
set "PYTHONPATH=%SRC%"
call :pyrun 4 "%ELO_PY%" "%PM%\build_ffbridge_postmortem_archive.py" --source-dir "%SOURCE%" --archive-dir "%CLUB_OUT%\postmortem_archive_flat" --hierarchical-dir "%HIER%" --hierarchical-layout-seed "%HIER%\metadata.json" --workers %ARCHIVE_WORKERS% --since %ARCHIVE_SINCE% --skip-compact
set "STEP_EXIT=%ERRORLEVEL%"
set "PYTHONPATH=%PYTHONPATH_SAVED%"
if not "%STEP_EXIT%"=="0" goto :error

:: ====================================================================
:: STAGE 5: PREDICTION MODELS
:: ACBL 5a/5b build a ~6,000-column matrix and join ACBL seat Elo.
:: FFBridge Club files are the narrow deal matrix, so 5a and 5b are one
:: step: join boards to hand features plus all-seat suit lengths from PBN,
:: attach each player's and partnership's record from earlier days,
:: synthesize Contract, temporal-split. Pre-session scratch Elo is attached
:: when elo_cache\*.results.parquet exists (session_id = tournament_id).
:: Feature code shared with the postmortem: ..\mlBridge\ffbridge_prediction_features.py.
:: ACBL checkpoints are not loaded.
:: 5c uses the ACBL venv because that interpreter has the CUDA PyTorch
:: build. The Elo venv does not.
:: ====================================================================
echo.
echo [Stage 5] Prediction models...
if not exist "%ACBL_PY%" (
    echo *** FAILED: ACBL venv not found: %ACBL_PY%
    echo Stage 5 needs PyTorch. Create it from ..\acbl-pipeline or install torch into that venv.
    set "STEP_LABEL=5"
    goto :error
)

:: ---- 5a/5b ----
:: READS:  E:\bridge\data\ffbridge\ffbridge_club_board_results_augmented.parquet
::         E:\bridge\data\ffbridge\ffbridge_club_hand_records_augmented_narrow.parquet
::         optional elo_cache\*.results.parquet (scratch Elo before the session)
:: WRITES: E:\bridge\data\ffbridge\ffbridge_prediction_data_train.parquet
::         E:\bridge\data\ffbridge\ffbridge_prediction_data_test.parquet
::         E:\bridge\data\ffbridge\ffbridge_prediction_data_meta.json
::         E:\bridge\data\ffbridge\SavedModels\ffbridge_prediction_player_history.parquet
::         E:\bridge\data\ffbridge\SavedModels\ffbridge_prediction_pair_history.parquet
::           (running player/partnership totals; the postmortem looks
::           players up in these, so they ship with the checkpoints)
:: TIME:   MEASURED 2026-10-18: 43s for 9.59M boards (history tables 4s,
::         as-of lookups 20s), 87 features. Date split at 2026-01-01.
::         --if-stale skips the rewrite when those outputs are newer than
::         the club parquets (and the Elo results file, when one is used).
echo   [5a/5b] Building prediction train/test data...
call :pyrun 5a "%ACBL_PY%" "%~dp0ffbridge_prediction_data.py" --club-dir "%CLUB_OUT%" --output-dir "%CLUB_OUT%" --if-stale
if errorlevel 1 goto :error

:: ---- 5c ----
:: READS:  the train/test parquets from 5a/5b
:: WRITES: E:\bridge\data\ffbridge\SavedModels\ffbridge_club_predicted_*_torch_model.pth
::         E:\bridge\data\ffbridge\SavedModels\*_schema.json
::         E:\bridge\data\ffbridge\ffbridge_prediction_metrics.json
::           (per-target accuracy/MAE, club-rank Spearman per club session
::           vs an Elo-gap line, and the joint contract/declarer pick)
:: TIME:   MEASURED 2026-10-20: 147s for three models
::         (Declarer_Direction, Contract, Pct_NS) on 7.37M train rows, CUDA
::         (270s on 2026-10-19). 32,000-row batches, 16 epochs with early
::         stopping. Per model: about 10s building shards, 20-32s training,
::         7-9s test evaluation. 8,192-row batches for 8 epochs took 144s of
::         training and scored 0.7 points lower on both classifiers.
::         --if-stale skips a target whose .pth is newer than the train parquet.
::         Shards are built in memory and moved to the GPU once per target;
::         no shard files are written.
:: ACCURACY 2026 test (2.43M boards): Declarer_Direction 0.6619 (joint 0.6647),
::         Contract 0.3871 (joint 0.3887), Pct_NS MAE 0.2316, club-rank
::         Spearman 0.529 vs Elo line 0.511 (12,105 club sessions).
:: PUBLISH: ..\elo\u.bat (run by ..\u_all.bat) copies SavedModels to src\data
::         and X1; restart the postmortem container to stage them.
echo   [5c] Training prediction models...
set MPLBACKEND=Agg
call :pyrun 5c "%ACBL_PY%" "%~dp0ffbridge_prediction_train.py" --data-dir "%CLUB_OUT%" --if-stale
if errorlevel 1 goto :error

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
echo.
echo  Postmortem archive (published by ..\elo\u.bat):
echo    postmortem_archive_hierarchical\manifest.parquet, metadata.json, fragments\
echo.
echo  SavedModels (published by ..\elo\u.bat):
echo    ffbridge_club_predicted_declarer_direction_torch_model.pth
echo    ffbridge_club_predicted_contract_torch_model.pth
echo    ffbridge_club_predicted_pct_ns_torch_model.pth
echo    ffbridge_prediction_player_history.parquet
echo    ffbridge_prediction_pair_history.parquet
echo ======================================================================
del /q "%STEP_OK%" 2>nul
goto :eof

:: usage: call :pyrun LABEL python.exe script.py [args...]
:: Use %~2 for the exe. "%2" double-quotes the already-quoted exe path,
:: and cmd then hands python.exe to another interpreter as a script.
:: Arguments are collected with shift: %10 and up do not exist in cmd.
:pyrun
set "STEP_LABEL=%~1"
set "PYEXE=%~2"
set "PYARGS="
shift
shift
:pyrun_args
if "%~1"=="" goto :pyrun_go
set PYARGS=%PYARGS% %1
shift
goto :pyrun_args
:pyrun_go
del /q "%STEP_OK%" 2>nul
call :now STEP_T0
"%PYEXE%" %PYARGS% && echo.>"%STEP_OK%"
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
