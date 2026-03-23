"""特徴量エンジニアリングのユニットテスト"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_race_df(n_races=3, n_courses=6):
    """テスト用の最小レースDataFrameを生成"""
    records = []
    for race_no in range(1, n_races + 1):
        for course in range(1, n_courses + 1):
            records.append({
                "date": "2026-01-01",
                "jyo_cd": "01",
                "race_no": race_no,
                "course": course,
                "racer_id": 1000 + (race_no - 1) * 6 + course,
                "win_rate": float(course),
                "nirenritsu": float(course * 10),
                "sanrenritsu": float(course * 15),
                "motor_nirenritsu": 35.0,
                "st": 0.15,
                "tenji_time": 6.75,
                "weather": "晴",
                "wind_speed": 3,
                "wave": 4,
                "water_temp": 14.0,
                "racer_class": "B1",
                "rank": course,  # course番目が course着
            })
    return pd.DataFrame(records)


class TestOpponentFeatures:
    def test_opponent_features_shape(self):
        """対戦相手特徴量が正しく生成されること"""
        from convert_boatracecsv import add_opponent_features
        df = make_race_df(n_races=2)
        result = add_opponent_features(df)
        expected_cols = [
            "opponent_avg_win_rate", "opponent_max_win_rate",
            "win_rate_vs_best", "opponent_avg_class", "relative_form_rank"
        ]
        for col in expected_cols:
            assert col in result.columns, f"列 '{col}' が見つかりません"

    def test_opponent_avg_win_rate_correct(self):
        """opponent_avg_win_rateがコース1に対して正しく計算されること"""
        from convert_boatracecsv import add_opponent_features
        df = make_race_df(n_races=1)
        result = add_opponent_features(df)
        # コース1の選手: win_rate=1.0, 対戦相手は win_rate=2,3,4,5,6 → 平均=4.0
        course1_row = result[result["course"] == 1].iloc[0]
        assert abs(course1_row["opponent_avg_win_rate"] - 4.0) < 1e-6

    def test_no_nan_in_opponent_features(self):
        """対戦相手特徴量にNaNがないこと"""
        from convert_boatracecsv import add_opponent_features
        df = make_race_df(n_races=3)
        result = add_opponent_features(df)
        cols = ["opponent_avg_win_rate", "opponent_max_win_rate", "win_rate_vs_best"]
        for col in cols:
            assert result[col].isna().sum() == 0, f"'{col}' にNaNがあります"


class TestStackingFeatures:
    def test_stacking_features_shape(self):
        """スタッキング特徴量が正しく生成されること"""
        from run_train55k import add_stacking_features
        df = make_race_df(n_races=2)
        df["prob_win"] = np.random.rand(len(df))
        result = add_stacking_features(df)
        expected_cols = ["prob_win_max_in_race", "prob_win_rank", "is_likely_winner"]
        for col in expected_cols:
            assert col in result.columns, f"列 '{col}' が見つかりません"

    def test_prob_win_rank_is_1_to_6(self):
        """prob_win_rankが1〜6の値を持つこと"""
        from run_train55k import add_stacking_features
        df = make_race_df(n_races=2)
        df["prob_win"] = np.linspace(0.05, 0.30, len(df))
        result = add_stacking_features(df)
        ranks = result["prob_win_rank"].unique()
        assert set(ranks).issubset(set(range(1, 7)))

    def test_is_likely_winner_uses_threshold(self):
        """is_likely_winner が WINNER_THRESHOLD 以上の艇にフラグを立てること"""
        from run_train55k import add_stacking_features, WINNER_THRESHOLD
        df = make_race_df(n_races=1)
        # コース1は0.10、コース6は0.40（閾値0.25より高い）
        probs = [0.10, 0.15, 0.20, 0.20, 0.15, 0.40]
        df["prob_win"] = probs * 1  # n_races=1
        result = add_stacking_features(df)
        # prob_win >= 0.25 の艇だけフラグが1
        expected = (df["prob_win"] >= WINNER_THRESHOLD).astype(int).tolist()
        assert result["is_likely_winner"].tolist() == expected
