# FFBridge pipeline

Sibling of `acbl-pipeline`. This folder orchestrates; it does not own ingest
or augmentation code.

`ffbridge_all.bat` refreshes the Lancelot / quality cache (`..\elo`), writes
Club-shaped BridgeStats parquets to `E:\bridge\data\ffbridge`, then publishes
through `..\bridgestats-ffbridge\u.bat`.

`ffbridge_recent.bat` downloads games newer than the quality parquet into
`E:\bridge\data\ffbridge\recent`. Schedule it with `hour`, `day`, `week`, or
`quarter`. Postmortem archive queries and the `recent_club_games` SQL view
read that file. The next `ffbridge_all.bat` quality build absorbs those
sessions and drops them from the recent file.

```bat
ffbridge_all.bat
ffbridge_recent.bat day
set FFBRIDGE_SESSION_LIMIT=50 && ffbridge_all.bat
```
