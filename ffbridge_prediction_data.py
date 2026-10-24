"""Build FFBridge train/test frames for contract, declarer, and score models.

ACBL stages 5a and 5b start from a ~6,000-column hand-record matrix and from
seat/pair Elo keyed like ACBL. FFBridge Club files are the narrow deal matrix
(about 40 board columns and 39 hand columns), and FFBridge Elo is a
pair-session table (``player1_id`` / ``player2_id``, scratch Elo *before* the
session). Trained ACBL weights cannot be reused. This script is the FFBridge
counterpart of those two stages:

- join each board to its deal features on PBN, plus suit lengths for all seats
- build ``Contract`` as ``{level}{strain}{double}{declarer}`` (or ``PASS``)
- attach each player's and partnership's record from earlier days
- attach pre-session scratch Elo when a results parquet is available
- write a temporal train/test split (default cutoff 2026-01-01)

Score rows, tricks, and the auction itself are not features. ``Contract`` and
``Declarer_Direction`` are targets only, so the percentage model does not see
the auction.

The percentage target is North-South's matchpoint percentage, not the
declarer's. The declarer is North-South at some tables and East-West at
others, so the declarer's percentage has no relation to the NS-EW strength
gap; NS percentage does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

_REPO_SRC = Path(__file__).resolve().parent.parent
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from mlBridge.ffbridge_prediction_features import (  # noqa: E402
    HISTORY_FEATURES,
    SHAPE_FEATURES,
    attach_history,
    build_history_tables,
    shape_features,
)


DEFAULT_CLUB_DIR = Path(r"E:\bridge\data\ffbridge")
DEFAULT_CUTOFF = "2026-01-01"
TRAIN_NAME = "ffbridge_prediction_data_train.parquet"
TEST_NAME = "ffbridge_prediction_data_test.parquet"
META_NAME = "ffbridge_prediction_data_meta.json"
# Staged with the models: the postmortem looks players up in these.
PLAYER_HISTORY_NAME = "ffbridge_prediction_player_history.parquet"
PAIR_HISTORY_NAME = "ffbridge_prediction_pair_history.parquet"

TARGETS = ("Declarer_Direction", "Contract", "Pct_NS")

# Deal facts known before the auction. SL_N_ML_SJ is an integer on the narrow
# hand file, not the ACBL suit-shape string, so it is renamed before training
# (mlBridge treats the original name as a categorical shape enum).
HAND_NUMERIC = (
    "ParScore",
    "DD_N_C",
    "DD_N_D",
    "DD_N_H",
    "DD_N_S",
    "DD_N_N",
    "SL_N_C",
    "SL_N_D",
    "SL_N_H",
    "SL_N_S",
    "SL_N_ML_SJ",
    "HCP_NS",
    "HCP_EW",
    "HCP_N",
    "HCP_E",
    "HCP_S",
    "HCP_W",
    "QT_N",
    "QT_E",
    "QT_S",
    "QT_W",
    "QT_NS",
    "QT_EW",
    "DP_N",
    "DP_N_C",
    "DP_N_D",
    "DP_N_H",
    "DP_N_S",
    "DP_NS",
    "DP_EW",
)
RENAMES = {"SL_N_ML_SJ": "SL_N_Longest"}
# These are contract-type words (Pass, Partial, Game, SSlam, GSlam), not a
# mlBridge categorical name. Stored as an ordered code so the trainer does
# not require a predefined enum for the column.
CT_FEATURES = ("CT_N_S", "CT_N_H", "CT_N_D", "CT_N_C", "CT_N_N")
CT_CODES = {"Pass": 0.0, "Partial": 1.0, "Game": 2.0, "SSlam": 3.0, "GSlam": 4.0}
CATEGORICAL_FEATURES = ("Dealer", "Vul")
BASE_ELO_FEATURES = ("Elo_N", "Elo_E", "Elo_S", "Elo_W", "Elo_NS", "Elo_EW")
# Missing flags are taken before the fill, the gap after it. Serving fills a
# missing rating with the schema mean, so training fills with the mean too.
DERIVED_ELO_FEATURES = ("Elo_Gap", "Elo_NS_Missing", "Elo_EW_Missing")
ELO_FEATURES = (*BASE_ELO_FEATURES, *DERIVED_ELO_FEATURES)
# Not features. Kept so pair-session ranks can be evaluated per club.
META_COLUMNS = ("session_id", "Club", "Date", "Board", "PBN", "Pair_Key_NS", "Pair_Key_EW")
# Below this fraction of rows with a North seat Elo, the join missed and the
# Elo columns would be almost entirely a fill value.
MIN_ELO_MATCH_RATE = 0.05


def feature_columns(frame: pl.DataFrame) -> list[str]:
    numeric = [RENAMES.get(name, name) for name in HAND_NUMERIC]
    present = [
        name
        for name in dict.fromkeys(
            (*numeric, *SHAPE_FEATURES, *CT_FEATURES, *CATEGORICAL_FEATURES, *ELO_FEATURES, *HISTORY_FEATURES)
        )
        if name in frame.columns
    ]
    return present


def _id_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.replace(r"\.0$", "")
        .str.strip_chars()
    )


def find_elo_results(explicit: str | None) -> Path | None:
    """Newest pair-session results parquet, or None when Elo has not been built."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Elo results parquet not found: {path}")
        return path
    env_file = os.environ.get("FFBRIDGE_ELO_RESULTS")
    if env_file and Path(env_file).is_file():
        return Path(env_file)
    roots = []
    cache_dir = os.environ.get("FFBRIDGE_CACHE_DIR")
    if cache_dir:
        roots.append(Path(cache_dir) / "elo_cache")
    # The deployed Elo and postmortem containers mount src/data at /data.
    # Local pipeline runs use the same canonical host tree.
    roots.append(Path(__file__).resolve().parent.parent / "data" / "ffbridge" / "elo_cache")
    roots.append(Path(__file__).resolve().parent.parent / "elo" / "data" / "ffbridge" / "elo_cache")
    roots.append(DEFAULT_CLUB_DIR / "elo_cache")
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(root.glob("*Lancelot_API*.results.parquet"))
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


# Floats are rounded to this many decimals before hashing. Polars 2 sums
# group_by aggregates in a varying order, so history features can differ in
# the last bit between builds of the same data (seen: 21 of 7.4M
# Hist_Pct_Gap values at 1e-16). That noise must not force a retrain.
HASH_FLOAT_DECIMALS = 9


def frame_content_hash(frame: pl.DataFrame) -> str:
    """SHA-256 of the column layout and every row, in order.

    Row order is included: the trainer's seeded shuffle permutes row
    positions, so a reordered frame trains a different model. Float columns
    are rounded to ``HASH_FLOAT_DECIMALS`` and -0.0 becomes 0.0 first.
    ``hash_rows`` is stable within one Polars version; an upgrade changes the
    hash and causes one retrain.
    """
    canonical = frame.with_columns(
        (pl.col(name).round(HASH_FLOAT_DECIMALS) + 0.0).alias(name)
        for name, dtype in frame.schema.items()
        if dtype.is_float()
    )
    digest = hashlib.sha256()
    digest.update(json.dumps([[name, str(dtype)] for name, dtype in frame.schema.items()]).encode())
    digest.update(canonical.hash_rows(seed=0).to_numpy().tobytes())
    return digest.hexdigest()


def previous_train_hash(meta_path: Path) -> str | None:
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8")).get("train_hash")


def _outputs_are_fresh(output_dir: Path, sources: list[Path]) -> bool:
    outputs = [output_dir / TRAIN_NAME, output_dir / TEST_NAME, output_dir / META_NAME]
    if not all(path.is_file() for path in outputs):
        return False
    # The train file is rewritten only when its rows change, so its mtime can
    # be older than the sources. Test and meta are written on every build.
    oldest_output = min(path.stat().st_mtime for path in outputs[1:])
    newest_source = max(path.stat().st_mtime for path in sources if path.is_file())
    return oldest_output >= newest_source


def build_labels() -> list[pl.Expr]:
    """Contract matches mlBridge ``Contract_enum``: level, strain, double, seat."""
    seat_ok = pl.col("Declarer_Direction").is_in(["N", "E", "S", "W"])
    suit_ok = pl.col("BidSuit").is_in(["C", "D", "H", "S", "N"])
    level_ok = pl.col("BidLvl").is_between(1, 7, closed="both")
    played = level_ok & suit_ok & seat_ok
    double = (
        pl.when(pl.col("Dbl").is_in(["X", "XX"]))
        .then(pl.col("Dbl"))
        .otherwise(pl.lit(""))
    )
    contract = (
        pl.when(played)
        .then(
            pl.col("BidLvl").cast(pl.Utf8)
            + pl.col("BidSuit")
            + double
            + pl.col("Declarer_Direction")
        )
        .otherwise(pl.lit("PASS"))
        .alias("Contract")
    )
    direction = (
        pl.when(played)
        .then(pl.col("Declarer_Direction"))
        .otherwise(pl.lit("NULL"))
        .alias("Declarer_Direction")
    )
    declarer_pct = pl.col("Declarer_Pct").cast(pl.Float64, strict=False)
    pct_ns = (
        pl.when(played & pl.col("Declarer_Direction").is_in(["N", "S"]))
        .then(declarer_pct)
        .when(played & pl.col("Declarer_Direction").is_in(["E", "W"]))
        .then(1.0 - declarer_pct)
        .otherwise(None)
        .alias("Pct_NS")
    )
    return [contract, direction, pct_ns]


def pair_key_columns() -> list[pl.Expr]:
    """Order-independent partnership keys, built before player ids are dropped."""
    exprs = []
    for name, left, right in (("Pair_Key_NS", "N", "S"), ("Pair_Key_EW", "E", "W")):
        a = _id_expr(f"Player_ID_{left}")
        b = _id_expr(f"Player_ID_{right}")
        exprs.append(
            pl.when(a.is_null() | b.is_null() | (a == "") | (b == ""))
            .then(None)
            .when(a <= b)
            .then(a + pl.lit("|") + b)
            .otherwise(b + pl.lit("|") + a)
            .alias(name)
        )
    return exprs


def _join_deals(club_dir: Path, max_rows: int | None, cutoff: date) -> pl.DataFrame:
    boards_path = club_dir / "ffbridge_club_board_results_augmented.parquet"
    hands_path = club_dir / "ffbridge_club_hand_records_augmented_narrow.parquet"
    if not boards_path.is_file():
        raise FileNotFoundError(f"Club board results not found: {boards_path}")
    if not hands_path.is_file():
        raise FileNotFoundError(f"Club hand records not found: {hands_path}")

    board_columns = [
        "session_id",
        "Club",
        "Date",
        "Board",
        "PBN",
        "Dealer",
        "Vul",
        "Declarer_Direction",
        "BidLvl",
        "BidSuit",
        "Dbl",
        "Declarer_Pct",
        "Player_ID_N",
        "Player_ID_E",
        "Player_ID_S",
        "Player_ID_W",
    ]
    hands = (
        pl.scan_parquet(hands_path)
        .select(["PBN", *HAND_NUMERIC, *CT_FEATURES])
        .unique(subset=["PBN"], keep="first")
        .collect()
    )
    hands = shape_features(hands).lazy()
    boards = pl.scan_parquet(boards_path).select(board_columns)
    joined = boards.join(hands, on="PBN", how="inner", maintain_order="left").with_columns(
        *build_labels(), *pair_key_columns()
    )
    joined = joined.drop("BidLvl", "BidSuit", "Dbl", "Declarer_Pct")
    if max_rows:
        train_part = joined.filter(pl.col("Date") < cutoff).head(max_rows)
        test_part = joined.filter(pl.col("Date") >= cutoff).head(max(1, max_rows // 4))
        frame = pl.concat([train_part.collect(), test_part.collect()], how="vertical")
    else:
        frame = joined.collect()
    return frame.rename(RENAMES)


def attach_scratch_elo(
    frame: pl.DataFrame, results_path: Path, cutoff: date
) -> tuple[pl.DataFrame, bool]:
    """Join pre-session scratch Elo onto the four seats and the two pairs.

    Uses Elo *before* the session only. After-session ratings are the result
    of these boards and would leak the target. ``scratch_pair_elo`` is the
    mean of the two *after* ratings, so the pair rating is rebuilt here from
    the two before ratings. The NS pair's ``club_id`` fills ``Club``.

    Missing ratings are filled with the mean over training rows
    (``Date < cutoff``) only, so new test sessions do not change train rows.
    """
    schema = pl.read_parquet_schema(results_path)
    required = {
        "tournament_id",
        "player1_id",
        "player2_id",
        "player1_scratch_elo_before",
        "player2_scratch_elo_before",
        "club_id",
    }
    missing = sorted(required - set(schema))
    if missing:
        print(f"Elo parquet {results_path.name} is missing {missing}; continuing without Elo.")
        return frame, False

    results = (
        pl.scan_parquet(results_path)
        .select(
            _id_expr("tournament_id").alias("session_id"),
            _id_expr("player1_id").alias("p1"),
            _id_expr("player2_id").alias("p2"),
            pl.col("player1_scratch_elo_before").cast(pl.Float32).alias("e1"),
            pl.col("player2_scratch_elo_before").cast(pl.Float32).alias("e2"),
            _id_expr("club_id").alias("club_id"),
        )
        .with_columns(((pl.col("e1") + pl.col("e2")) / 2.0).alias("pair_elo"))
        .collect()
    )
    players = pl.concat(
        [
            results.select("session_id", pl.col("p1").alias("player_id"), pl.col("e1").alias("elo")),
            results.select("session_id", pl.col("p2").alias("player_id"), pl.col("e2").alias("elo")),
        ]
    ).filter(
        pl.col("player_id").is_not_null()
        & (pl.col("player_id") != "")
        & (pl.col("player_id") != "None")
    )
    players = players.group_by(["session_id", "player_id"]).agg(pl.col("elo").mean())
    pairs = (
        results.select(
            "session_id",
            pl.min_horizontal("p1", "p2").alias("a"),
            pl.max_horizontal("p1", "p2").alias("b"),
            "pair_elo",
            "club_id",
        )
        .group_by(["session_id", "a", "b"])
        .agg(pl.col("pair_elo").mean(), pl.col("club_id").drop_nulls().first())
    )

    out = frame.with_columns(_id_expr("session_id").alias("_sid"))
    height = out.height
    for seat in ("N", "E", "S", "W"):
        lookup = (
            out.select(pl.col("_sid").alias("session_id"), _id_expr(f"Player_ID_{seat}").alias("player_id"))
            .with_row_index("_row")
            .join(players, on=["session_id", "player_id"], how="left")
            .select("_row", pl.col("elo").alias(f"Elo_{seat}"))
        )
        out = out.with_row_index("_row").join(lookup, on="_row", how="left").drop("_row")
        if out.height != height:
            raise RuntimeError(f"Elo seat join for {seat} changed the row count")

    for pair_name, left, right in (("NS", "N", "S"), ("EW", "E", "W")):
        keys = out.select(
            pl.col("_sid").alias("session_id"),
            pl.min_horizontal(_id_expr(f"Player_ID_{left}"), _id_expr(f"Player_ID_{right}")).alias("a"),
            pl.max_horizontal(_id_expr(f"Player_ID_{left}"), _id_expr(f"Player_ID_{right}")).alias("b"),
        ).with_row_index("_row")
        lookup = keys.join(pairs, on=["session_id", "a", "b"], how="left").select(
            "_row",
            pl.col("pair_elo").alias(f"Elo_{pair_name}"),
            pl.col("club_id").alias(f"_club_{pair_name}"),
        )
        out = out.with_row_index("_row").join(lookup, on="_row", how="left").drop("_row")
        if out.height != height:
            raise RuntimeError(f"Elo pair join for {pair_name} changed the row count")
    out = out.with_columns(
        pl.coalesce(pl.col("Club").cast(pl.Utf8), pl.col("_club_NS"), pl.col("_club_EW")).alias("Club")
    ).drop("_club_NS", "_club_EW")
    print(f"Club known on {out['Club'].is_not_null().mean():.1%} of rows")

    matched = out.filter(pl.col("Elo_N").is_not_null()).height
    rate = matched / height if height else 0.0
    print(f"Elo match rate (North seat, session_id = tournament_id): {matched:,}/{height:,} ({rate:.1%})")
    player_ids = [f"Player_ID_{seat}" for seat in "NESW"]
    if rate < MIN_ELO_MATCH_RATE:
        print("Elo match rate is below 5%; dropping Elo columns so they are not trained as a fill value.")
        return out.drop("_sid", *BASE_ELO_FEATURES, *player_ids), False

    out = out.with_columns(
        pl.col("Elo_NS").is_null().cast(pl.Float32).alias("Elo_NS_Missing"),
        pl.col("Elo_EW").is_null().cast(pl.Float32).alias("Elo_EW_Missing"),
    )
    is_train = pl.col("Date") < cutoff
    for column in BASE_ELO_FEATURES:
        mean = out.select(pl.col(column).filter(is_train).mean()).item()
        fill = 0.0 if mean is None else float(mean)
        out = out.with_columns(pl.col(column).fill_null(fill).fill_nan(fill).cast(pl.Float32))
    out = out.with_columns((pl.col("Elo_NS") - pl.col("Elo_EW")).alias("Elo_Gap"))
    return out.drop("_sid", *player_ids), True


def _attach_player_history(club_dir: Path, models_dir: Path, frame: pl.DataFrame) -> pl.DataFrame:
    """Player and partnership records from boards before each session's date.

    The running-total tables are built from every club board and written to
    ``models_dir`` so the postmortem can look up the same records.
    """
    started = datetime.now()
    boards = pl.scan_parquet(club_dir / "ffbridge_club_board_results_augmented.parquet")
    players, pairs = build_history_tables(boards)
    models_dir.mkdir(parents=True, exist_ok=True)
    players.write_parquet(models_dir / PLAYER_HISTORY_NAME)
    pairs.write_parquet(models_dir / PAIR_HISTORY_NAME)
    print(
        f"History tables: {players.height:,} player-days, {pairs.height:,} partnership-days "
        f"({(datetime.now() - started).total_seconds():.0f}s)"
    )
    out = attach_history(frame, players, pairs, {seat: f"Player_ID_{seat}" for seat in "NESW"})
    known = out.select((pl.col("Hist_Boards_N") > 0).mean()).item()
    print(f"North has earlier boards on {known:.1%} of rows ({(datetime.now() - started).total_seconds():.0f}s)")
    return out


def _ct_code_expr(name: str) -> pl.Expr:
    return (
        pl.col(name)
        .cast(pl.Utf8)
        .replace_strict(CT_CODES, default=None)
        .cast(pl.Float32)
        .fill_null(0.0)
        .alias(name)
    )


def _cast_model_columns(frame: pl.DataFrame) -> pl.DataFrame:
    numeric = [RENAMES.get(name, name) for name in HAND_NUMERIC]
    numeric = [
        name
        for name in dict.fromkeys((*numeric, *SHAPE_FEATURES, *ELO_FEATURES, *HISTORY_FEATURES))
        if name in frame.columns
    ]
    exprs = [
        pl.col(name).cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).alias(name)
        for name in numeric
    ]
    for name in CT_FEATURES:
        if name in frame.columns:
            exprs.append(_ct_code_expr(name))
    for name in CATEGORICAL_FEATURES:
        if name in frame.columns:
            exprs.append(pl.col(name).cast(pl.Utf8).alias(name))
    exprs.append(pl.col("Pct_NS").cast(pl.Float32, strict=False))
    exprs.append(pl.col("Contract").cast(pl.Utf8))
    exprs.append(pl.col("Declarer_Direction").cast(pl.Utf8))
    return frame.with_columns(exprs)


def build_prediction_data(
    club_dir: Path,
    output_dir: Path,
    cutoff: date,
    elo_results: Path | None,
    max_rows: int | None,
) -> None:
    print("[5a] Joining club boards to narrow deal features...")
    frame = _join_deals(club_dir, max_rows, cutoff)
    print(f"Joined rows: {frame.height:,}  columns: {len(frame.columns)}")
    frame = _attach_player_history(club_dir, output_dir / "SavedModels", frame)
    used_elo = False
    if elo_results is not None:
        print(f"[5a] Attaching pre-session scratch Elo from {elo_results}")
        frame, used_elo = attach_scratch_elo(frame, elo_results, cutoff)
    else:
        print("[5a] No Elo results parquet found; models will use deal features only.")
        player_ids = [f"Player_ID_{seat}" for seat in "NESW" if f"Player_ID_{seat}" in frame.columns]
        if player_ids:
            frame = frame.drop(player_ids)
    frame = _cast_model_columns(frame)

    print(f"[5b] Splitting on Date >= {cutoff.isoformat()} ...")
    train = frame.filter(pl.col("Date") < cutoff)
    test = frame.filter(pl.col("Date") >= cutoff)
    if train.height == 0 or test.height == 0:
        raise SystemExit(
            f"Temporal split is empty (train={train.height:,}, test={test.height:,}, cutoff={cutoff.isoformat()})."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / TRAIN_NAME
    test_path = output_dir / TEST_NAME
    meta_path = output_dir / META_NAME
    train_hash = frame_content_hash(train)
    train_kept = train_path.is_file() and previous_train_hash(meta_path) == train_hash
    if train_kept:
        print(f"Train rows unchanged (hash {train_hash[:12]}); keeping {train_path.name}.")
    else:
        train.write_parquet(train_path)
    test.write_parquet(test_path)
    features = feature_columns(train)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cutoff": cutoff.isoformat(),
        "train_rows": train.height,
        "train_hash": train_hash,
        "test_rows": test.height,
        "features": features,
        "targets": list(TARGETS),
        "elo_results": str(elo_results) if used_elo and elo_results else None,
        "max_rows": max_rows,
        "contract": "level + strain + double + declarer, or PASS",
        "score_target": "Pct_NS",
        "history_tables": [PLAYER_HISTORY_NAME, PAIR_HISTORY_NAME],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"{'Kept' if train_kept else 'Wrote'} {train_path.name} ({train.height:,} rows); "
        f"wrote {test_path.name} ({test.height:,} rows), {len(features)} features."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FFBridge prediction train/test parquets.")
    parser.add_argument("--club-dir", type=Path, default=DEFAULT_CLUB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CLUB_DIR)
    parser.add_argument("--test-cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--elo-results", default=None, help="Pair-session Elo results parquet. Auto-detected when omitted.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap each side. For smoke tests; not used by ffbridge_all.bat.")
    parser.add_argument("--if-stale", action="store_true")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cutoff = date.fromisoformat(args.test_cutoff)
    elo_results = find_elo_results(args.elo_results)
    sources = [
        args.club_dir / "ffbridge_club_board_results_augmented.parquet",
        args.club_dir / "ffbridge_club_hand_records_augmented_narrow.parquet",
    ]
    if elo_results is not None:
        sources.append(elo_results)
    if args.if_stale and not args.max_rows and _outputs_are_fresh(args.output_dir, sources):
        print(f"Prediction train/test parquets are newer than their sources in {args.output_dir}.")
        return 0
    build_prediction_data(args.club_dir, args.output_dir, cutoff, elo_results, args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
