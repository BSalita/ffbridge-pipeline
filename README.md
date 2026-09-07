# FFBridge pipeline

Sibling of `acbl-pipeline`. This folder orchestrates; it does not own ingest
or augmentation code.

`ffbridge_all.bat` refreshes the Lancelot / quality cache (`..\elo`), writes
Club-shaped BridgeStats parquets to `E:\bridge\data\ffbridge`, then publishes
through `..\bridgestats-ffbridge\u.bat`.

```bat
ffbridge_all.bat
set FFBRIDGE_SESSION_LIMIT=50 && ffbridge_all.bat
```
