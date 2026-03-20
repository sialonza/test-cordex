#!/usr/bin/env python3
"""
v2_features.py
実データからローリング特徴量を構築する。

データリーク防止: 各レースの特徴量は、そのレース以前のデータのみで計算。
  - レーサー過去成績 (累積 + 直近N回)
  - モーター成績 (累積, 場別)
  - コース別・場別統計
  - レース内相対指標

出力: data/processed/v2_train.csv, v2_val.csv, v2_test.csv, v2_feature_cols.txt
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def load_data():
    path = os.path.join(RAW_DIR, "race_results_real.csv")
    df = pd.read_csv(path, dtype={"jyo_cd": str})
    df["jyo_cd"] = df["jyo_cd"].str.zfill(2)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "jyo_cd", "race_no", "course"]).reset_index(drop=True)
    print(f"ロード: {len(df):,} 行 ({df['date'].min().date()} ~ {df['date'].max().date()})")
    return df


# ──────────────────────────────────────────────────────────────
# ローリング特徴量 (データリーク防止済み)
# ──────────────────────────────────────────────────────────────

def _shifted_cumstats(df, group_col, value_col, prefix):
    """
    group_col でグループ化し、value_col の累積統計を計算。
    現在行を含まないようシフト。
    """
    g = df.groupby(group_col)[value_col]
    cumsum = g.cumsum() - df[value_col]
    cumcount = g.cumcount()  # 0-indexed (= number of previous rows)
    safe_count = cumcount.clip(lower=1)

    df[f"{prefix}_cum_mean"] = cumsum / safe_count
    df[f"{prefix}_cum_count"] = cumcount
    # 最初のレース (履歴なし) は NaN
    df.loc[cumcount == 0, f"{prefix}_cum_mean"] = np.nan
    return df


def racer_features(df):
    """レーサーの過去成績から特徴量を計算"""
    print("  レーサー特徴量...")
    df["is_win"] = (df["rank"] == 1).astype(float)
    df["is_top2"] = (df["rank"] <= 2).astype(float)
    df["is_top3"] = (df["rank"] <= 3).astype(float)
    df["rank_inv"] = 1.0 / df["rank"].clip(lower=1)  # 着順の逆数 (1着=1.0, 6着=0.167)

    # --- 累積統計 (全履歴) ---
    for val, pref in [
        ("is_win", "racer_win"),
        ("is_top3", "racer_top3"),
        ("rank", "racer_rank"),
        ("st", "racer_st"),
    ]:
        df = _shifted_cumstats(df, "racer_id", val, pref)

    # --- 直近N回の成績 (EWM = 指数加重移動平均) ---
    # EWMはrecent form（最近の調子）を捉える最良の方法
    for span, suffix in [(10, "_ew10"), (30, "_ew30")]:
        for val, pref in [("is_win", "racer_win"), ("rank_inv", "racer_inv")]:
            ewm_col = f"{pref}{suffix}"
            # groupby + EWM + shift(1) でリーク防止
            df[ewm_col] = (
                df.groupby("racer_id")[val]
                .transform(lambda x: x.ewm(span=span, min_periods=1).mean().shift(1))
            )

    # --- コース別累積勝率 ---
    for c in range(1, 7):
        mask = df["course"] == c
        col = f"racer_c{c}_win"
        df[col] = np.nan
        if mask.any():
            sub = df.loc[mask].copy()
            g = sub.groupby("racer_id")["is_win"]
            cumsum = g.cumsum() - sub["is_win"]
            cumcount = g.cumcount()
            sub[col] = cumsum / cumcount.clip(lower=1)
            sub.loc[cumcount == 0, col] = np.nan
            df.loc[mask, col] = sub[col]

    # コース別勝率を1列にまとめる (自分のコースの値)
    df["racer_course_win_rate"] = np.nan
    for c in range(1, 7):
        col = f"racer_c{c}_win"
        mask = df["course"] == c
        df.loc[mask, "racer_course_win_rate"] = df.loc[mask, col]
    # 個別コース列は削除
    df.drop(columns=[f"racer_c{c}_win" for c in range(1, 7)], inplace=True)

    # --- 場別累積勝率 ---
    df = _shifted_cumstats(
        df.assign(_key=df["racer_id"].astype(str) + "_" + df["jyo_cd"]),
        "_key", "is_win", "racer_jyo_win"
    )
    df.drop(columns=["_key"], inplace=True)

    return df


def motor_features(df):
    """モーター成績 (場×モーター)"""
    print("  モーター特徴量...")
    df["_mkey"] = df["jyo_cd"] + "_" + df["motor_no"].astype(str)
    df = _shifted_cumstats(df, "_mkey", "is_win", "motor_win")
    df = _shifted_cumstats(df, "_mkey", "is_top2", "motor_top2")
    df = _shifted_cumstats(df, "_mkey", "rank", "motor_rank")
    df.drop(columns=["_mkey"], inplace=True)
    return df


def venue_features(df):
    """場×コース別の統計"""
    print("  場別特徴量...")
    df["_vkey"] = df["jyo_cd"] + "_" + df["course"].astype(str)
    df = _shifted_cumstats(df, "_vkey", "is_win", "venue_course_win")
    df.drop(columns=["_vkey"], inplace=True)
    return df


def race_relative_features(df):
    """レース内相対指標 (同一レース6艇の中での順位・偏差)"""
    print("  レース内相対特徴量...")
    race_key = ["date", "jyo_cd", "race_no"]

    # 各指標のレース内順位
    for col, ascending, new_col in [
        ("racer_win_cum_mean", False, "rr_win_rank"),
        ("racer_st_cum_mean", True, "rr_st_rank"),
        ("racer_top3_cum_mean", False, "rr_top3_rank"),
        ("motor_win_cum_mean", False, "rr_motor_rank"),
        ("tenji_time", True, "rr_tenji_rank"),
    ]:
        if col in df.columns:
            df[new_col] = df.groupby(race_key)[col].rank(
                ascending=ascending, method="min"
            )

    # レース平均との差分
    for col, new_col in [
        ("racer_win_cum_mean", "rr_win_vs_avg"),
        ("racer_st_cum_mean", "rr_st_vs_avg"),
        ("tenji_time", "rr_tenji_vs_avg"),
    ]:
        if col in df.columns:
            df[new_col] = df[col] - df.groupby(race_key)[col].transform("mean")

    # レース内最強レーサーとの差
    if "racer_win_cum_mean" in df.columns:
        df["rr_win_vs_best"] = (
            df["racer_win_cum_mean"]
            - df.groupby(race_key)["racer_win_cum_mean"].transform("max")
        )

    # レース内のレーサー数 (6でないケース: 欠場)
    df["rr_n_boats"] = df.groupby(race_key)["course"].transform("count")

    return df


def interaction_features(df):
    """交互作用特徴量"""
    print("  交互作用特徴量...")
    # コース有利度 × レーサー実力
    win_mean = df["racer_win_cum_mean"].fillna(0.167)
    course_power = (7 - df["course"]) / 6.0  # 1コース=1.0, 6コース=0.167
    df["ix_course_x_racer"] = course_power * win_mean

    # モーター × レーサー
    motor_mean = df["motor_win_cum_mean"].fillna(0.167)
    df["ix_motor_x_racer"] = motor_mean * win_mean

    # ST × コース (内枠で遅いSTはペナルティ)
    st_mean = df["racer_st_cum_mean"].fillna(0.15)
    df["ix_st_x_course"] = st_mean * df["course"]

    # 展示タイム × レーサー実力
    df["ix_tenji_x_racer"] = df["tenji_time"].fillna(6.75) * win_mean

    return df


def calendar_features(df):
    """日付・カレンダー特徴量"""
    print("  カレンダー特徴量...")
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def weather_features(df):
    """天候エンコード"""
    print("  天候特徴量...")
    weather_map = {"晴": 0, "曇り": 1, "雨": 2, "雪": 3}
    df["weather_num"] = df["weather"].map(weather_map).fillna(0)
    return df


def define_features():
    """使用する特徴量のリスト"""
    return [
        # 基本
        "course",
        # レーサー累積
        "racer_win_cum_mean", "racer_top3_cum_mean",
        "racer_rank_cum_mean", "racer_st_cum_mean",
        "racer_win_cum_count",
        # レーサー直近 (EWM)
        "racer_win_ew10", "racer_win_ew30",
        "racer_inv_ew10", "racer_inv_ew30",
        # コース別・場別
        "racer_course_win_rate",
        "racer_jyo_win_cum_mean",
        # モーター
        "motor_win_cum_mean", "motor_top2_cum_mean", "motor_rank_cum_mean",
        "motor_win_cum_count",
        # 場×コース
        "venue_course_win_cum_mean",
        # 展示タイム・ST (当日データ)
        "st", "tenji_time",
        # レース内相対
        "rr_win_rank", "rr_st_rank", "rr_top3_rank",
        "rr_motor_rank", "rr_tenji_rank",
        "rr_win_vs_avg", "rr_st_vs_avg", "rr_tenji_vs_avg",
        "rr_win_vs_best", "rr_n_boats",
        # 交互作用
        "ix_course_x_racer", "ix_motor_x_racer",
        "ix_st_x_course", "ix_tenji_x_racer",
        # カレンダー
        "month", "dayofweek", "is_weekend",
        # 天候
        "weather_num", "wind_speed", "wave",
    ]


def split_data(df, feature_cols):
    """時系列分割: 70% / 15% / 15%"""
    print("データ分割...")
    # ウォームアップ期間を除外 (最初90日はローリング特徴量が不安定)
    min_date = df["date"].min()
    warmup_cutoff = min_date + pd.Timedelta(days=90)
    df_valid = df[df["date"] >= warmup_cutoff].copy()
    print(f"  ウォームアップ除外: {min_date.date()} ~ {warmup_cutoff.date()} ({len(df) - len(df_valid):,} 行)")

    n = len(df_valid)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df_valid.iloc[:train_end]
    val = df_valid.iloc[train_end:val_end]
    test = df_valid.iloc[val_end:]

    print(f"  Train: {len(train):,} ({train['date'].min().date()} ~ {train['date'].max().date()})")
    print(f"  Val:   {len(val):,} ({val['date'].min().date()} ~ {val['date'].max().date()})")
    print(f"  Test:  {len(test):,} ({test['date'].min().date()} ~ {test['date'].max().date()})")

    return train, val, test


def main():
    print("=" * 60)
    print("v2 特徴量エンジニアリング (ローリング・リークフリー)")
    print("=" * 60)

    df = load_data()

    # ターゲット
    df["target_win"] = (df["rank"] == 1).astype(int)
    df["target_2nd"] = (df["rank"] == 2).astype(int)
    df["target_top3"] = (df["rank"] <= 3).astype(int)

    # 特徴量生成
    df = racer_features(df)
    df = motor_features(df)
    df = venue_features(df)
    df = race_relative_features(df)
    df = interaction_features(df)
    df = calendar_features(df)
    df = weather_features(df)

    feature_cols = define_features()
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  警告: 欠損特徴量: {missing}")

    print(f"\n有効特徴量: {len(available)}個")

    # 欠損値処理
    for c in available:
        if df[c].isna().any():
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val)

    # 分割・保存
    train, val, test = split_data(df, available)

    # 保存するカラム
    meta_cols = ["date", "jyo_cd", "jyo_name", "race_no", "course",
                 "racer_id", "rank", "boat"]
    target_cols = ["target_win", "target_2nd", "target_top3"]
    save_cols = [c for c in meta_cols if c in df.columns] + available + target_cols

    train[save_cols].to_csv(os.path.join(PROC_DIR, "v2_train.csv"), index=False)
    val[save_cols].to_csv(os.path.join(PROC_DIR, "v2_val.csv"), index=False)
    test[save_cols].to_csv(os.path.join(PROC_DIR, "v2_test.csv"), index=False)

    with open(os.path.join(PROC_DIR, "v2_feature_cols.txt"), "w") as f:
        f.write("\n".join(available))

    print(f"\n保存完了: {PROC_DIR}/v2_*.csv")
    print(f"特徴量数: {len(available)}")

    # サンプル表示
    print("\n特徴量サンプル (test先頭):")
    print(test[available[:10]].head(6).to_string())
    print("\n完了!")


if __name__ == "__main__":
    main()
