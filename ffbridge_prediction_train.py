"""Train FFBridge declarer, contract, and percentage models.

Uses the same mlBridge shard trainer as ACBL stage 5c, on the narrow deal
features written by ``ffbridge_prediction_data.py``. ACBL checkpoints are not
loaded: the column layout is different.

``Pct_NS`` is also scored the way the postmortem uses it: predictions are
averaged per pair-session and compared with actual pair percentages by rank
correlation within each club session, both raw and centered on each board.
An Elo-gap line fitted on the training rows is the baseline the model has
to beat.

Default network sizes are smaller than ACBL's because this matrix is tens of
columns, not thousands. Pass ``--epochs`` to match a longer ACBL run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

_SRC = Path(__file__).resolve().parent
_REPO_SRC = _SRC.parent
# The mlBridge directory itself must not go on sys.path: it contains
# mlBridge.py, and that file would shadow the package.
for _path in (_SRC, _REPO_SRC):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

import polars as pl

from ffbridge_prediction_data import (
    CATEGORICAL_FEATURES,
    CT_FEATURES,
    META_NAME,
    TARGETS,
    TEST_NAME,
    TRAIN_NAME,
    _ct_code_expr,
    feature_columns,
)


SAVED_DIR_NAME = "SavedModels"


def _encode_contract_types(frame: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for name in CT_FEATURES:
        if name not in frame.columns:
            continue
        if frame.schema[name].is_numeric():
            exprs.append(pl.col(name).cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0))
        else:
            exprs.append(_ct_code_expr(name))
    if exprs:
        frame = frame.with_columns(exprs)
    return frame


def _model_name(target: str) -> str:
    return f"ffbridge_club_predicted_{target.lower()}_torch_model"


def _hyperparameters(target: str) -> dict:
    # Narrow inputs. ACBL's 2048-wide nets are sized for ~6,000 columns.
    # 32,000-row batches with twice the 8,192-row learning rates and 16 epochs:
    # 2026-10-20 benchmark, 96s vs 144s training, both classifiers +0.7 points,
    # Pct_NS MAE and club-rank Spearman unchanged.
    if target == "Declarer_Direction":
        return {
            "layers": [512, 256, 128],
            "dropout": 0.10,
            "lr": 2e-3,
            "weight_decay": 0.0,
            "bs": 32000,
            "y_range": None,
        }
    if target == "Contract":
        return {
            "layers": [512, 256, 128],
            "dropout": 0.20,
            "lr": 6e-4,
            "weight_decay": 0.0,
            "bs": 32000,
            "y_range": None,
        }
    return {
        "layers": [256, 128, 64],
        "dropout": 0.05,
        "lr": 6e-4,
        "weight_decay": 1e-4,
        "bs": 32000,
        "y_range": None,
    }


def _cast_categoricals(frame: pl.DataFrame, target: str) -> pl.DataFrame:
    from mlBridge.mlBridgeLib import (
        Contract_enum,
        Dealer_enum,
        Declarer_Direction_enum,
        Vul_enum,
    )

    exprs = []
    if "Dealer" in frame.columns:
        exprs.append(pl.col("Dealer").cast(Dealer_enum, strict=False))
    if "Vul" in frame.columns:
        exprs.append(pl.col("Vul").cast(Vul_enum, strict=False))
    if target == "Declarer_Direction":
        exprs.append(pl.col(target).cast(Declarer_Direction_enum, strict=False))
    elif target == "Contract":
        exprs.append(pl.col(target).cast(Contract_enum, strict=False))
    else:
        exprs.append(pl.col(target).cast(pl.Float32, strict=False))
    if exprs:
        frame = frame.with_columns(exprs)
    return frame.filter(pl.col(target).is_not_null())


def prepare_target_frame(frame: pl.DataFrame, target: str) -> pl.DataFrame:
    """Keep deal features and this target. Drop the other targets and ids."""
    columns = feature_columns(frame)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Training frame is missing features: {missing}")
    out = frame.select([*columns, target])
    out = _encode_contract_types(out)
    if target == "Pct_NS":
        out = out.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    out = _cast_categoricals(out, target)
    out = out.filter(pl.col(target).is_not_null())
    numeric = [name for name in out.columns if name not in CATEGORICAL_FEATURES and name != target]
    if numeric:
        out = out.with_columns(
            [
                pl.col(name).cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0)
                for name in numeric
                if out.schema[name].is_numeric()
            ]
        )
    return out


def _fresh_model(models_dir: Path, target: str, train_path: Path) -> bool:
    checkpoint = models_dir / f"{_model_name(target)}.pth"
    if not checkpoint.is_file():
        return False
    return checkpoint.stat().st_mtime >= train_path.stat().st_mtime


def _score(actual: pl.Series, predicted: pl.Series, target: str) -> dict:
    if target == "Pct_NS":
        both = pl.DataFrame({"y": actual.cast(pl.Float64), "p": predicted.cast(pl.Float64)}).drop_nulls()
        error = (both["p"] - both["y"]).abs()
        y_std = float(both["y"].std() or 0.0)
        p_std = float(both["p"].std() or 0.0)
        return {
            "rows": both.height,
            "mae": float(error.mean()) if both.height else None,
            "mae_constant_half": float((both["y"] - 0.5).abs().mean()) if both.height else None,
            "variance_ratio": (p_std / y_std) if y_std else None,
        }
    both = pl.DataFrame(
        {"y": actual.cast(pl.Utf8), "p": predicted.cast(pl.Utf8)}
    ).drop_nulls()
    matches = both.filter(pl.col("y") == pl.col("p")).height
    return {
        "rows": both.height,
        "accuracy": (matches / both.height) if both.height else None,
    }


MIN_PAIR_BOARDS = 8
MIN_CLUB_PAIRS = 4


def elo_gap_line(train_df: pl.DataFrame) -> tuple[float, float]:
    """Least-squares ``Pct_NS = intercept + slope * Elo_Gap`` on the training rows."""
    rows = train_df.filter(pl.col("Pct_NS").is_not_null() & pl.col("Pct_NS").is_finite())
    stats = rows.select(
        pl.col("Elo_Gap").cast(pl.Float64).mean().alias("mx"),
        pl.col("Pct_NS").cast(pl.Float64).mean().alias("my"),
        pl.cov(pl.col("Elo_Gap").cast(pl.Float64), pl.col("Pct_NS").cast(pl.Float64)).alias("cxy"),
        pl.col("Elo_Gap").cast(pl.Float64).var().alias("vx"),
    ).row(0, named=True)
    if not stats["vx"]:
        raise ValueError("Elo_Gap has no variance in the training rows")
    slope = stats["cxy"] / stats["vx"]
    return stats["my"] - slope * stats["mx"], slope


def center_per_board(column: str) -> pl.Expr:
    """Shift each club board to a 50% mean, the way matchpoints average."""
    board = ["session_id", "Club", "Board"]
    return (pl.col(column) - pl.col(column).mean().over(board) + 0.5).clip(0.0, 1.0)


def pair_session_rank_quality(frame: pl.DataFrame, columns: list[str]) -> dict:
    """Spearman correlation of predicted and actual pair percentage per club session.

    North-South pairs score ``Pct_NS``; East-West pairs score its complement on
    the same rows. Both directions are ranked together, as ``Club_Rank`` is.
    """
    def side_score(name: str, east_west: bool) -> pl.Expr:
        return (1.0 - pl.col(name)) if east_west else pl.col(name)

    pieces = []
    for key, east_west in (("Pair_Key_NS", False), ("Pair_Key_EW", True)):
        pieces.append(
            frame.filter(pl.col(key).is_not_null())
            .group_by(["session_id", "Club", key])
            .agg(
                side_score("Pct_NS", east_west).mean().alias("actual"),
                *[side_score(name, east_west).mean().alias(name) for name in columns],
                pl.len().alias("boards"),
            )
            .filter(pl.col("boards") >= MIN_PAIR_BOARDS)
            .drop(key)
        )
    pairs = pl.concat(pieces, how="vertical")
    clubs = (
        pairs.group_by(["session_id", "Club"])
        .agg(
            pl.len().alias("pairs"),
            *[
                pl.corr(name, "actual", method="spearman").alias(name)
                for name in columns
            ],
        )
        .filter(pl.col("pairs") >= MIN_CLUB_PAIRS)
    )
    result = {"club_sessions": clubs.height, "pairs": int(clubs["pairs"].sum())}
    for name in columns:
        result[name] = {
            "mean_spearman": float(clubs[name].fill_nan(None).mean()),
            "pair_mae": float((pairs[name] - pairs["actual"]).abs().mean()),
        }
    result["constant_half_pair_mae"] = float((pairs["actual"] - 0.5).abs().mean())
    return result


def evaluate_pct_ns_ranking(
    test_rows: pl.DataFrame,
    predicted: pl.Series,
    line: tuple[float, float],
) -> dict:
    intercept, slope = line
    frame = test_rows.with_columns(predicted.cast(pl.Float64).alias("model_raw")).filter(
        pl.col("Club").is_not_null()
    )
    frame = frame.with_columns(
        (intercept + slope * pl.col("Elo_Gap").cast(pl.Float64)).clip(0.0, 1.0).alias("elo_line_raw"),
    )
    frame = frame.with_columns(
        center_per_board("model_raw").alias("model_per_board"),
        center_per_board("elo_line_raw").alias("elo_line_per_board"),
    )
    quality = pair_session_rank_quality(
        frame,
        ["model_raw", "model_per_board", "elo_line_raw", "elo_line_per_board"],
    )
    quality["elo_line"] = {"intercept": intercept, "slope_per_100_elo": slope * 100.0}
    return quality


def train_one(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target: str,
    models_dir: Path,
    epochs: int,
    device: str,
    shard_rows: int,
) -> dict:
    import math
    import time

    from mlBridge.mlBridgeAiLib import (
        build_torch_shards,
        generate_and_save_schema,
        predict_model,
        train_model_from_shards,
    )

    started = time.perf_counter()
    hp = _hyperparameters(target)
    model_name = _model_name(target)
    print(f"\n{'=' * 60}\n{model_name}  device={device}  epochs={epochs}  layers={hp['layers']}")
    working = prepare_target_frame(train_df, target)
    print(f"Training rows: {working.height:,}  features: {working.width - 1}")
    if working.height < 2:
        raise SystemExit(f"Not enough training rows for {target}: {working.height}")

    schema = generate_and_save_schema(
        working,
        models_dir,
        model_name,
        target,
        layers=hp["layers"],
        dropout=hp["dropout"],
        apply_scaling_parameters=True,
        y_range=hp["y_range"],
        verbose=False,
    )
    rows = working.height
    shard_count = math.ceil(rows / shard_rows) if shard_rows else 0
    effective_shards = shard_rows if shard_count >= 2 else max(1, rows // 2)
    shards = build_torch_shards(working, schema, shard_rows_count=effective_shards, apply_scaling=True)
    del working
    prepared = time.perf_counter()
    train_model_from_shards(
        schema,
        epochs=epochs,
        bs=hp["bs"],
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        use_amp=device == "cuda",
        device=device,
        seed=42,
        early_stop_patience=3,
        verbose=True,
        resident_data=True,
        shards=shards,
    )
    del shards
    trained = time.perf_counter()

    test_rows = test_df
    if target == "Pct_NS":
        test_rows = test_df.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    evaluation = prepare_target_frame(test_rows, target)
    features = evaluation.drop(target)
    predicted = predict_model(
        models_dir,
        model_name,
        features,
        device=device,
        strict_validation=True,
        parity_drop_non_finite=False,
    )
    pred_column = f"{target}_Pred"
    if pred_column not in predicted.columns:
        raise RuntimeError(f"predict_model did not return {pred_column}; columns={predicted.columns}")
    if predicted.height != evaluation.height:
        raise RuntimeError(
            f"Prediction row count {predicted.height:,} != test rows {evaluation.height:,} for {target}"
        )
    metrics = _score(evaluation[target], predicted[pred_column], target)
    if target == "Pct_NS":
        if evaluation.height != test_rows.height:
            raise RuntimeError(
                f"Pct_NS evaluation rows {evaluation.height:,} != test rows {test_rows.height:,}"
            )
        metrics["ranking"] = evaluate_pct_ns_ranking(
            test_rows, predicted[pred_column], elo_gap_line(train_df)
        )
    metrics.update(
        {
            "target": target,
            "model": model_name,
            "train_rows": rows,
            "features": working_feature_count(schema),
            "device": device,
            "epochs": epochs,
            "layers": hp["layers"],
        }
    )
    print(f"{target} test metrics: {metrics}")
    print(
        f"{target} elapsed: prepare {prepared - started:.1f}s, train {trained - prepared:.1f}s, "
        f"evaluate {time.perf_counter() - trained:.1f}s"
    )
    return metrics


def schema_classes(models_dir: Path, model_name: str) -> list[str]:
    schema = json.loads((models_dir / f"{model_name}_schema.json").read_text(encoding="utf-8"))
    class_to_idx = schema["class_to_idx"]
    return sorted(class_to_idx, key=class_to_idx.get)


JOINT_SAMPLE_ROWS = 300_000


def _probs_matrix(series: pl.Series, width: int):
    return series.list.to_array(width).to_numpy()


def evaluate_joint_contract_declarer(test_df: pl.DataFrame, models_dir: Path, device: str) -> dict:
    """Accuracy of each classifier alone and of one contract chosen from both.

    Scored on a fixed sample: full class probabilities for every test row
    would not fit in memory.
    """
    from mlBridge.ffbridge_prediction_features import joint_contract_and_declarer
    from mlBridge.mlBridgeAiLib import predict_model

    rows = test_df.filter(pl.col("Contract").is_not_null())
    if rows.height > JOINT_SAMPLE_ROWS:
        rows = rows.sample(JOINT_SAMPLE_ROWS, seed=42)
    evaluation = prepare_target_frame(rows, "Contract")
    if evaluation.height != rows.height:
        raise RuntimeError("Contract and declarer test rows do not line up")
    features = evaluation.drop("Contract")
    actual_contract = evaluation["Contract"].cast(pl.Utf8)
    actual_direction = rows["Declarer_Direction"].cast(pl.Utf8)
    outputs = {}
    for target in ("Contract", "Declarer_Direction"):
        name = _model_name(target)
        predicted = predict_model(
            models_dir, name, features, device=device, return_probs=True,
            strict_validation=True, parity_drop_non_finite=False,
        )
        outputs[target] = (schema_classes(models_dir, name), predicted)
    contract_classes, contract_out = outputs["Contract"]
    direction_classes, direction_out = outputs["Declarer_Direction"]
    joint_contract, joint_direction = joint_contract_and_declarer(
        contract_classes,
        _probs_matrix(contract_out["Contract_Probs"], len(contract_classes)),
        direction_classes,
        _probs_matrix(direction_out["Declarer_Direction_Probs"], len(direction_classes)),
    )
    alone_contract = contract_out["Contract_Pred"].cast(pl.Utf8)
    alone_direction = direction_out["Declarer_Direction_Pred"].cast(pl.Utf8)
    frame = pl.DataFrame(
        {
            "contract": actual_contract,
            "direction": actual_direction,
            "alone_contract": alone_contract,
            "alone_direction": alone_direction,
            "joint_contract": pl.Series(joint_contract, dtype=pl.Utf8),
            "joint_direction": pl.Series(joint_direction, dtype=pl.Utf8),
        }
    ).with_columns(
        pl.when(pl.col("alone_contract") == "PASS").then(pl.lit("NULL"))
        .otherwise(pl.col("alone_contract").str.slice(-1)).alias("contract_model_direction")
    )
    stats = frame.select(
        (pl.col("alone_contract") == pl.col("contract")).mean().alias("contract_alone"),
        (pl.col("joint_contract") == pl.col("contract")).mean().alias("contract_joint"),
        (pl.col("alone_direction") == pl.col("direction")).mean().alias("direction_alone"),
        (pl.col("contract_model_direction") == pl.col("direction")).mean().alias("direction_from_contract_model"),
        (pl.col("joint_direction") == pl.col("direction")).mean().alias("direction_joint"),
        (pl.col("contract_model_direction") == pl.col("alone_direction")).mean().alias("alone_models_agree"),
    ).row(0, named=True)
    result = {"target": "Joint_Contract_Declarer", "rows": frame.height, **stats}
    print(f"Joint contract/declarer metrics: {result}")
    return result


def working_feature_count(schema: dict) -> int:
    numeric = schema.get("numerical_feature_cols") or []
    categorical = schema.get("categorical_feature_cols") or []
    return len(numeric) + len(categorical)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train FFBridge prediction models.")
    parser.add_argument("--data-dir", type=Path, default=Path(r"E:\bridge\data\ffbridge"))
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--target", action="append", choices=TARGETS, dest="targets")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--shard-rows", type=int, default=500_000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--if-stale", action="store_true")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from datetime import datetime

    started = datetime.now()
    print(f"[ffbridge-prediction-train] start {started:%Y-%m-%dT%H:%M:%S}")
    status = _run(args)
    ended = datetime.now()
    print(f"[ffbridge-prediction-train] end {ended:%Y-%m-%dT%H:%M:%S} (elapsed {(ended - started).total_seconds():.1f}s)")
    return status


def _run(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed in this interpreter. "
            "ffbridge_all.bat uses acbl-pipeline\\.venv, which has the CUDA build."
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")

    train_path = args.data_dir / TRAIN_NAME
    test_path = args.data_dir / TEST_NAME
    if not train_path.is_file() or not test_path.is_file():
        raise SystemExit(
            f"Missing {train_path.name} or {test_path.name}. Run ffbridge_prediction_data.py first."
        )
    models_dir = args.models_dir or (args.data_dir / SAVED_DIR_NAME)
    models_dir.mkdir(parents=True, exist_ok=True)
    targets = list(args.targets) if args.targets else list(TARGETS)

    pending = []
    for target in targets:
        if args.if_stale and _fresh_model(models_dir, target, train_path):
            print(f"Keeping {_model_name(target)}.pth; it is newer than {train_path.name}.")
            continue
        pending.append(target)
    if not pending:
        print("Prediction models are up to date.")
        return 0

    print(f"Loading {train_path.name}...")
    train_df = pl.read_parquet(train_path)
    print(f"Loading {test_path.name}...")
    test_df = pl.read_parquet(test_path)

    all_metrics = []
    metrics_path = args.data_dir / "ffbridge_prediction_metrics.json"
    if metrics_path.is_file():
        try:
            all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            all_metrics = []
    by_target = {item.get("target"): item for item in all_metrics if isinstance(item, dict)}

    for target in pending:
        by_target[target] = train_one(
            train_df,
            test_df,
            target,
            models_dir,
            epochs=args.epochs,
            device=device,
            shard_rows=args.shard_rows,
        )
    classifiers = ("Contract", "Declarer_Direction")
    if any(name in pending for name in classifiers) and all(
        (models_dir / f"{_model_name(name)}.pth").is_file() for name in classifiers
    ):
        by_target["Joint_Contract_Declarer"] = evaluate_joint_contract_declarer(test_df, models_dir, device)
    ordered = [by_target[name] for name in (*TARGETS, "Joint_Contract_Declarer") if name in by_target]
    metrics_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    meta_path = args.data_dir / META_NAME
    print(f"Wrote {metrics_path}")
    if meta_path.is_file():
        print(f"Feature list: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
