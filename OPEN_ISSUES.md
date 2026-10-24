# FFBridge pipeline: open issues

Left over from the 2026-10-24 performance work. Everything done that day is
on `master` and pushed in `mlBridge`, `postmortem-ffbridge`,
`bridgestats-ffbridge` and `ffbridge-pipeline`, and was published with
`u_all.bat`. Numbers below are from that day's full run (1 h 17 min, mostly a
436-session stage 4 backlog) unless stated.

## Do next

### Restart the morty_chat server
Stopped on 2026-10-24 so `MortyBridgeChatbot\.venv` could move to DuckDB 2.0.0
(it was running as `MortyBridgeChatbot\.venv\Scripts\python.exe -m morty_chat`).
The venv upgrade is done and its 783 tests pass; the server was not restarted.

### Restart the postmortem container
`u_all.bat` copies the retrained stage 5 models (`SavedModels`) and the new
archive sessions to `src\data` and X1. The postmortem container stages
`SavedModels` only when it restarts.

## Speed

### Stage 1: download from FFBridge is serial (biggest daily cost)
10 new sessions took 8 min 11 s to fetch 2,625 ranking/team-score files, out
of a 10 min 19 s stage; building them took 89 s.
- Where: `elo\ffbridge_quality_pipeline.py`, `fetch_missing_artifacts` and
  the team-score fetch (around line 600; it does not pass the shared HTTP
  session, so connections are not reused).
- Ideas: fetch with a small bounded thread pool (be polite to the API);
  reuse one `requests.Session`; cache 404s with a TTL so missing files are
  not re-requested every day (failed fetches currently retry up to 6 times
  with backoff); start session discovery paging from the newest cached date
  instead of re-paging the whole year.

### Stage 4: fixed cost after the session loop
In the 436-session run the loop ended at 64 min but the stage took 72 min 40 s:
about 8 min for the last commit batch plus `archive.rebuild_indexes`, which
prints nothing while it runs. Profile `rebuild_indexes` (it reads the
manifest and fragment schemas).

### Smaller items from the 2026-10-24 review
- Stage 2 runs the full cache audit (all session JSONs plus the 171 MB
  training parquet) before its `--if-stale` check; a cheap signature check
  first would save 10-40 s on runs that skip.
- 5a as-of history lookups (~20 s) could run on unique (player, date) and
  (pair, date) keys, about 20x fewer rows, then join back.
- Stages 3, 4 and 5 do not depend on each other; stage 4 could run in the
  background beside them.
- `ffbridge_all.bat :now` starts PowerShell for every timestamp (~14 times).
- Stage 1 re-reads all ~7k fragments and rewrites every output daily
  (measured 25 s for that whole path; an incremental append would save about
  15 s but the current board file is not identical to a fresh rebuild, so it
  would need a rebuild as its base first). Low priority.

## Data

### 501 host clubs have no name
Simultaneous sessions now get each pair's host club from the ranking's
`simultaneousId` (it matches the Elo `club_id` for 99.99% of pairs). Of the
616 club codes now in the board file, 501 never run their own sessions, so
nothing names them and `ffbridge_clubs.parquet` lists only the 124 named
clubs. Needs a club-name source (an FFBridge club directory), or add the
codes with empty names if the UI copes with that.

### 714 board rows still have no Club
After the simultaneous-club fill, 714 of 9.86M board rows have no Club: pairs
not found in their session's ranking. Not investigated.

## Robustness

### ffbridge_all.bat stage 3 depends on the current-folder search
Stage 3 does `pushd "%STATS%"` then `call u.bat`. A shell with
`NoDefaultCurrentDirectoryInExePath=1` (Claude Code sets it) cannot find
`u.bat`, so stage 3 fails with "'u.bat' is not recognized". Use
`call "%STATS%\u.bat"`. Stage 3 also prints no `TIME[3]` line because it is
not run through `:pyrun`.

### Stage 4 smaller bugs (from the review)
- The result's `skipped_existing` is the archive's total size, not the number
  of sessions skipped.
- A worker `ValueError` other than "Malformed score row" aborts the whole run;
  prepared but uncommitted fragments (up to 49) are redone next run.

### Repository housekeeping
- `bridgestats-ffbridge\data\*.parquet` are tracked demo files that `u.bat`
  overwrites with the real outputs, so they always show as modified. Untrack
  them or point `u.bat` elsewhere.
- Stale worktree entries print "Permission denied" on every checkout:
  `mlBridge\.git\worktrees\mlBridge` and
  `bridgestats-ffbridge\.git\worktrees\_wt_builder`. Remove them by hand
  (`git worktree prune` cannot delete them while OneDrive holds them).
- `acbl-pipeline` branch `perf/pipeline-quick-wins` (4 commits) is unmerged;
  it belongs to the ACBL performance work.
- Scratch folders from 2026-10-24 testing, if still present:
  `E:\claude_sd`, `E:\claude_sd$1`, `E:\claude_m5`.
