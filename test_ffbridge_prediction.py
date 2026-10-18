import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

import ffbridge_prediction_data as data
import ffbridge_prediction_train as train
from mlBridge import ffbridge_prediction_features as features


class LabelTests(unittest.TestCase):
    def test_pct_ns_is_the_north_south_score_whoever_declared(self):
        frame = pl.DataFrame(
            {
                "Declarer_Direction": ["N", "E", "S", "W", "N"],
                "BidLvl": [3, 4, 1, 2, None],
                "BidSuit": ["N", "S", "C", "H", None],
                "Dbl": ["", "X", "", "", ""],
                "Declarer_Pct": [0.7, 0.7, 0.2, 0.2, 0.5],
            }
        ).with_columns(data.build_labels())
        self.assertEqual(frame["Contract"].to_list(), ["3NN", "4SXE", "1CS", "2HW", "PASS"])
        pct = [None if value is None else round(value, 6) for value in frame["Pct_NS"].to_list()]
        self.assertEqual(pct, [0.7, 0.3, 0.2, 0.8, None])

    def test_pair_keys_do_not_depend_on_seat_order(self):
        frame = pl.DataFrame(
            {
                "Player_ID_N": ["9", "3", None],
                "Player_ID_S": ["3", "9", "4"],
                "Player_ID_E": ["5", "5", "5"],
                "Player_ID_W": ["6", "6", "6"],
            }
        ).with_columns(data.pair_key_columns())
        self.assertEqual(frame["Pair_Key_NS"].to_list(), ["3|9", "3|9", None])
        self.assertEqual(frame["Pair_Key_EW"].to_list(), ["5|6", "5|6", "5|6"])


class EloTests(unittest.TestCase):
    def test_missing_flags_are_taken_before_the_mean_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elo.results.parquet"
            pl.DataFrame(
                {
                    "tournament_id": ["1"] * 4,
                    "player1_id": ["1", "3", "5", "7"],
                    "player2_id": ["2", "4", "6", "8"],
                    "player1_scratch_elo_before": [1600.0, 1400.0, 1500.0, 1500.0],
                    "player2_scratch_elo_before": [1600.0, 1200.0, 1500.0, 1500.0],
                    # After-session pair ratings must not be used.
                    "scratch_pair_elo": [9999.0, 9999.0, 9999.0, 9999.0],
                    "club_id": ["c1", "c1", "c2", "c2"],
                }
            ).write_parquet(path)
            boards = pl.DataFrame(
                {
                    "session_id": ["1"] * 20,
                    "Club": [None] * 20,
                    "Player_ID_N": ["1"] * 19 + ["99"],
                    "Player_ID_S": ["2"] * 19 + ["98"],
                    "Player_ID_E": ["3"] * 20,
                    "Player_ID_W": ["4"] * 20,
                }
            )
            out, used = data.attach_scratch_elo(boards, path)
        self.assertTrue(used)
        self.assertEqual(out["Elo_NS_Missing"].to_list(), [0.0] * 19 + [1.0])
        self.assertEqual(out["Elo_EW_Missing"].to_list(), [0.0] * 20)
        self.assertAlmostEqual(out["Elo_NS"][0], 1600.0)
        self.assertAlmostEqual(out["Elo_EW"][0], 1300.0)
        self.assertAlmostEqual(out["Elo_NS"][19], 1600.0)
        self.assertAlmostEqual(out["Elo_Gap"][0], 300.0)
        self.assertEqual(out["Club"].to_list(), ["c1"] * 20)


class RankingEvaluationTests(unittest.TestCase):
    def test_per_board_centering_removes_a_board_subset_artifact(self):
        rows = []
        # Board 1 leans NS by 0.2 at every table; EW pairs 3 and 4 skip it.
        for board, tables in ((1, (1, 2)), (2, (1, 2, 3, 4)), (3, (1, 2, 3, 4))):
            for table in tables:
                rows.append(
                    {
                        "session_id": "s",
                        "Club": "c",
                        "Board": board,
                        "Pair_Key_NS": f"n{table}",
                        "Pair_Key_EW": f"e{table}",
                        "Pct_NS": 0.5 + (0.05 if table == 1 else -0.05 if table == 4 else 0.0),
                        "pred": (0.7 if board == 1 else 0.5) + (0.01 if table == 1 else -0.01 if table == 4 else 0.0),
                    }
                )
        frame = pl.DataFrame(rows)
        frame = frame.with_columns(train.center_per_board("pred").alias("centered"))
        old_min = train.MIN_PAIR_BOARDS
        train.MIN_PAIR_BOARDS = 1
        try:
            quality = train.pair_session_rank_quality(frame, ["pred", "centered"])
        finally:
            train.MIN_PAIR_BOARDS = old_min
        self.assertEqual(quality["club_sessions"], 1)
        self.assertGreater(
            quality["centered"]["mean_spearman"],
            quality["pred"]["mean_spearman"],
        )

    def test_elo_gap_line_recovers_the_slope(self):
        frame = pl.DataFrame(
            {"Elo_Gap": [-200.0, 0.0, 200.0], "Pct_NS": [0.45, 0.5, 0.55]}
        )
        intercept, slope = train.elo_gap_line(frame)
        self.assertAlmostEqual(intercept, 0.5)
        self.assertAlmostEqual(slope * 100.0, 0.025)


class FeatureTests(unittest.TestCase):
    def test_history_uses_only_days_before_the_session(self):
        boards = pl.DataFrame(
            {
                "Date": [date(2026, 1, 5), date(2026, 1, 5), date(2026, 2, 1)],
                "session_id": ["1", "1", "2"],
                "Declarer_Direction": ["N", "W", "S"],
                "Declarer_Pct": [0.8, 0.6, 0.1],
                "Player_ID_N": ["10", "10", "10"],
                "Player_ID_S": ["20", "20", "21"],
                "Player_ID_E": ["30", "30", "30"],
                "Player_ID_W": ["40", "40", "40"],
            }
        )
        players, pairs = features.build_history_tables(boards.lazy())
        sessions = pl.DataFrame(
            {
                "Date": [date(2026, 1, 5), date(2026, 2, 1), date(2026, 3, 1)],
                "N": ["10", "10", "10"],
                "S": ["20", "21", "20"],
                "E": ["30", "30", "30"],
                "W": ["40", "40", "40"],
            }
        )
        out = features.attach_history(sessions, players, pairs, {seat: seat for seat in "NESW"})
        self.assertEqual(out["Hist_Boards_N"][0], 0.0)
        self.assertAlmostEqual(out["Hist_Pct_N"][0], 0.5)
        # Day one: N scored 0.8 and 0.4 (West declared at 0.6).
        self.assertAlmostEqual(out["Hist_Pct_N"][1], (1.2 + 30.0) / 62.0, places=5)
        self.assertAlmostEqual(out["Hist_Decl_Pct_N"][1], (0.8 + 7.5) / 16.0, places=5)
        self.assertAlmostEqual(out["Hist_Def_Pct_N"][1], (0.4 + 15.0) / 31.0, places=5)
        # 10|21 first played on 2026-02-01, so it has no record that day.
        self.assertEqual(out["Hist_Partner_Sessions_NS"][1], 0.0)
        self.assertAlmostEqual(out["Hist_Partner_Sessions_NS"][2], math.log1p(1.0), places=5)
        self.assertAlmostEqual(out["Hist_Boards_N"][2], math.log1p(3.0), places=5)

    def test_suit_lengths_and_fits_from_pbn(self):
        frame = pl.DataFrame(
            {"PBN": ["N:AKQJT.98.765.43 98.765.43.AKQJT2 765.43.AKQJT2.98 432.AKQJT2.98.765"]}
        )
        out = features.shape_features(frame).row(0, named=True)
        self.assertEqual((out["SL_N_S"], out["SL_N_H"], out["SL_N_D"], out["SL_N_C"]), (5.0, 2.0, 3.0, 2.0))
        self.assertEqual(out["SL_E_C"], 6.0)
        self.assertEqual(out["Fit_NS_S"], 8.0)
        self.assertEqual(out["Fit_EW_H"], 9.0)

    def test_joint_pick_keeps_contract_and_declarer_consistent(self):
        contracts = ["3NE", "3NN", "4SS", "PASS"]
        directions = ["E", "N", "NULL", "S", "W"]
        contract_probs = np.array([[0.40, 0.35, 0.20, 0.05], [0.1, 0.1, 0.1, 0.7]])
        direction_probs = np.array([[0.10, 0.60, 0.05, 0.20, 0.05], [0.3, 0.3, 0.1, 0.2, 0.1]])
        picked, seats = features.joint_contract_and_declarer(
            contracts, contract_probs, directions, direction_probs
        )
        self.assertEqual(picked, ["3NN", "PASS"])
        self.assertEqual(seats, ["N", "NULL"])

    def test_declarer_model_without_a_pass_out_class_is_neutral_on_pass(self):
        contracts = ["3NN", "3NS", "PASS"]
        directions = ["E", "N", "S", "W"]
        # A declarer model with no opinion leaves the contract model's choice.
        uniform = np.full((2, 4), 0.25)
        contract_probs = np.array([[0.30, 0.20, 0.50], [0.45, 0.15, 0.40]])
        picked, seats = features.joint_contract_and_declarer(contracts, contract_probs, directions, uniform)
        self.assertEqual(picked, ["PASS", "3NN"])
        self.assertEqual(seats, ["NULL", "N"])
        # A confident South moves the first board from PASS to 3NS.
        confident = np.array([[0.0, 0.05, 0.95, 0.0], [0.25, 0.25, 0.25, 0.25]])
        picked, _ = features.joint_contract_and_declarer(contracts, contract_probs, directions, confident)
        self.assertEqual(picked, ["3NS", "3NN"])
        # At zero weight the declarer model has no say.
        picked, _ = features.joint_contract_and_declarer(
            contracts, contract_probs, directions, confident, declarer_weight=0.0
        )
        self.assertEqual(picked, ["PASS", "3NN"])


if __name__ == "__main__":
    unittest.main()
