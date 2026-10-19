#!/usr/bin/env python3
"""Score FFBridge Pct_NS predictions the way acbl_prediction_rank_eval.py scores ACBL.

FFBridge test rows carry no section id, so the club session
(``session_id`` + ``Club``) stands in for an ACBL section. Pairs are ranked
against the other pairs in the same club session and direction, and compared
with the actual ranks by Spearman correlation. Variants match ACBL:

- ``model_raw``: the model output
- ``model_section_centered``: shifted so each club session averages 50%
- ``model_board_centered``: shifted so each board in a club session averages 50%
- ``elo_line_*``: the Elo-gap line fitted on the training rows

The ranking ``ffbridge_prediction_train.py`` reports (both directions ranked
together per club session) is recomputed first, from the same predictions, so
the two groupings can be compared and the row alignment checked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_SRC = Path(__file__).resolve().parent
for _path in (_SRC, _SRC.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import polars as pl  # noqa: E402

from ffbridge_prediction_data import TEST_NAME, TRAIN_NAME  # noqa: E402
from ffbridge_prediction_train import (  # noqa: E402
    SAVED_DIR_NAME,
    _model_name,
    elo_gap_line,
    evaluate_pct_ns_ranking,
    prepare_target_frame,
)

MIN_PAIR_BOARDS = 8
MIN_SECTION_PAIRS = 4
SECTION = ["session_id", "Club"]
DIRECTIONS = (("NS", "Pair_Key_NS", False), ("EW", "Pair_Key_EW", True))


def center(column: str, over: list[str]) -> pl.Expr:
    return (pl.col(column) - pl.col(column).mean().over(over) + 0.5).clip(0.0, 1.0)


def section_rank_quality(frame: pl.DataFrame, columns: list[str]) -> dict:
    """Spearman of predicted and actual pair percentage per club session and direction."""
    pieces = []
    for direction, key, east_west in DIRECTIONS:
        def side(name: str) -> pl.Expr:
            return (1.0 - pl.col(name)) if east_west else pl.col(name)

        pieces.append(
            frame.filter(pl.col(key).is_not_null())
            .group_by([*SECTION, key])
            .agg(
                side("Pct_NS").mean().alias("actual"),
                *[side(name).mean().alias(name) for name in columns],
                pl.len().alias("boards"),
            )
            .filter(pl.col("boards") >= MIN_PAIR_BOARDS)
            .drop(key)
            .with_columns(pl.lit(direction).alias("direction"))
        )
    pairs = pl.concat(pieces, how="vertical")
    groups = (
        pairs.group_by([*SECTION, "direction"])
        .agg(
            pl.len().alias("pairs"),
            *[pl.corr(name, "actual", method="spearman").alias(name) for name in columns],
        )
        .filter(pl.col("pairs") >= MIN_SECTION_PAIRS)
    )
    if groups.is_empty():
        raise ValueError(
            f"No club session direction has {MIN_SECTION_PAIRS}+ pairs with {MIN_PAIR_BOARDS}+ boards"
        )
    kept = pairs.join(groups.select(*SECTION, "direction"), on=[*SECTION, "direction"], how="semi")
    result = {
        "section_directions": groups.height,
        "pairs": kept.height,
        "median_pairs_per_section_direction": float(groups["pairs"].median()),
    }
    for name in columns:
        result[name] = {
            "mean_spearman": float(groups[name].fill_nan(None).mean()),
            "pair_mae": float((kept[name] - kept["actual"]).abs().mean()),
        }
    result["constant_half_pair_mae"] = float((kept["actual"] - 0.5).abs().mean())
    return result


def board_mean_offset(frame: pl.DataFrame) -> float:
    """Mean distance of each club board's average actual Pct_NS from 50%.

    Zero when percentages are matchpointed within the club session, as ACBL
    sections are. Larger when they are scored against a wider field.
    """
    return float(
        frame.group_by([*SECTION, "Board"])
        .agg(pl.col("Pct_NS").mean().alias("mean"))
        .select((pl.col("mean") - 0.5).abs().mean())
        .item()
    )


def predict_pct_ns(test_rows: pl.DataFrame, models_dir: Path, device: str) -> pl.Series:
    from mlBridge.mlBridgeAiLib import predict_model

    evaluation = prepare_target_frame(test_rows, "Pct_NS")
    if evaluation.height != test_rows.height:
        raise RuntimeError(f"Pct_NS evaluation rows {evaluation.height:,} != test rows {test_rows.height:,}")
    predicted = predict_model(
        models_dir,
        _model_name("Pct_NS"),
        evaluation.drop("Pct_NS"),
        device=device,
        strict_validation=True,
        parity_drop_non_finite=False,
    )
    if predicted.height != test_rows.height:
        raise RuntimeError(f"Prediction rows {predicted.height:,} != test rows {test_rows.height:,}")
    return predicted["Pct_NS_Pred"].cast(pl.Float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACBL-style section ranking evaluation for FFBridge Pct_NS.")
    parser.add_argument("--data-dir", type=Path, default=Path(r"E:\bridge\data\ffbridge"))
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    started = datetime.now()
    print(f"Start {started:%Y-%m-%d %H:%M:%S}")
    models_dir = args.models_dir or (args.data_dir / SAVED_DIR_NAME)
    line = elo_gap_line(pl.read_parquet(args.data_dir / TRAIN_NAME, columns=["Elo_Gap", "Pct_NS"]))
    test_rows = pl.read_parquet(args.data_dir / TEST_NAME).filter(
        pl.col("Pct_NS").is_not_null() & pl.col("Pct_NS").is_finite()
    )
    print(f"Predicting Pct_NS on {test_rows.height:,} test rows")
    predicted = predict_pct_ns(test_rows, models_dir, args.device)
    print(f"Predicted ({(datetime.now() - started).total_seconds():.0f}s)")

    metrics = {"combined_directions": evaluate_pct_ns_ranking(test_rows, predicted, line)}
    intercept, slope = line
    frame = (
        test_rows.select(*SECTION, "Board", "Pair_Key_NS", "Pair_Key_EW", "Pct_NS", "Elo_Gap")
        .with_columns(
            pl.col("Pct_NS").cast(pl.Float64),
            predicted.alias("model_raw"),
            (intercept + slope * pl.col("Elo_Gap").cast(pl.Float64)).clip(0.0, 1.0).alias("elo_line_raw"),
        )
        .filter(pl.col("Club").is_not_null())
        .with_columns(
            center("model_raw", SECTION).alias("model_section_centered"),
            center("model_raw", [*SECTION, "Board"]).alias("model_board_centered"),
            center("elo_line_raw", [*SECTION, "Board"]).alias("elo_line_board_centered"),
        )
    )
    columns = [
        "model_raw",
        "model_section_centered",
        "model_board_centered",
        "elo_line_raw",
        "elo_line_board_centered",
    ]
    quality = section_rank_quality(frame, columns)
    quality["board_mae"] = {name: float((frame[name] - frame["Pct_NS"]).abs().mean()) for name in columns}
    quality["board_mae"]["constant_half"] = float((frame["Pct_NS"] - 0.5).abs().mean())
    quality["club_board_mean_offset"] = board_mean_offset(frame)
    quality["elo_line"] = {"intercept": intercept, "slope_per_100_elo": slope * 100.0}
    quality["test_rows"] = frame.height
    metrics["section_direction"] = quality
    print(json.dumps(metrics, indent=2))

    out = args.data_dir / "ffbridge_prediction_rank_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ended = datetime.now()
    print(f"Wrote {out}\nEnd {ended:%Y-%m-%d %H:%M:%S} (elapsed {(ended - started).total_seconds():.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
