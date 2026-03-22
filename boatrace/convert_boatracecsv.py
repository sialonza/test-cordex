#!/usr/bin/env python3
"""
convert_boatracecsv.py
ダウンロードしたCSVデータを機械学習用に変換する。
- 特徴量エンジニアリング
- 追加データのマージ
- 学習/検証/テスト分割
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXTRA_DIR = os.path.join(DATA_DIR, "extra")
PROC_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def add_payout_market_features(df, payouts, windows=(90, 180)):
    """
    過去N日の会場×コース別 平均2連単払戻 から市場確率を推定する。

    payouts.rank1_course が「このコースが1着になった時の2連単払戻」を示す。
    レースごとに過去N日分の平均払戻を計算し、市場暗示確率 = 100/avg_payout を追加する。

    データリーク防止: rolling(closed='left') で当日以前のデータのみ使用。
    """
    print("  市場払戻特徴量を計算中...")

    payouts = payouts.copy()
    payouts["date"]   = pd.to_datetime(payouts["date"])
    payouts["jyo_cd"] = payouts["jyo_cd"].astype(str).str.zfill(2)
    payouts = payouts.rename(columns={"rank1_course": "course"})

    df = df.copy()
    df["date"]   = pd.to_datetime(df["date"])
    df["jyo_cd"] = df["jyo_cd"].astype(str).str.zfill(2)

    feat_cols = []
    for w in windows:
        feat_cols += [f"mkt_avg_payout_{w}d", f"mkt_win_prob_{w}d"]

    # ── Step 1: (jyo_cd, course) ごとに rolling 移動平均を計算 ──────────────
    stat_parts = []
    for (jyo, course), grp in payouts.groupby(["jyo_cd", "course"]):
        grp = grp.sort_values("date").set_index("date")
        row = pd.DataFrame(index=grp.index)
        row["jyo_cd"] = jyo
        row["course"] = int(course)
        for w in windows:
            # closed='left': [date-w, date) → 当日のレースを除外
            row[f"mkt_avg_payout_{w}d"] = (
                grp["payout_2nd"]
                .rolling(f"{w}D", min_periods=3, closed="left")
                .mean()
            )
        stat_parts.append(row.reset_index())

    stats_df = (pd.concat(stat_parts, ignore_index=True)
                  .sort_values(["jyo_cd", "course", "date"])
                  .reset_index(drop=True))

    # ── Step 2: merge_asof で race_results に結合 ────────────────────────────
    # 同一グループ (jyo_cd, course) ごとに "当日以前の最新 stat" を結合
    df = df.sort_values(["jyo_cd", "course", "date"]).reset_index(drop=True)

    merged_parts = []
    for (jyo, course), df_grp in df.groupby(["jyo_cd", "course"]):
        stat_grp = stats_df[
            (stats_df["jyo_cd"] == jyo) & (stats_df["course"] == int(course))
        ][["date"] + [f"mkt_avg_payout_{w}d" for w in windows]].sort_values("date")

        if len(stat_grp) == 0:
            merged_parts.append(df_grp)
            continue

        merged = pd.merge_asof(
            df_grp.sort_values("date"),
            stat_grp,
            on="date",
            direction="backward",   # 当日以前の最新値を使用
        )
        merged_parts.append(merged)

    result = (pd.concat(merged_parts, ignore_index=True)
                .sort_values(["date", "jyo_cd", "race_no", "course"])
                .reset_index(drop=True))

    # ── Step 3: 市場暗示確率を計算 ───────────────────────────────────────────
    for w in windows:
        payout_col = f"mkt_avg_payout_{w}d"
        prob_col   = f"mkt_win_prob_{w}d"
        # 払戻最低 200 円でクリップして除算 (NaN はそのまま)
        result[prob_col] = 100.0 / result[payout_col].clip(lower=200)

    print(f"    追加特徴量: {feat_cols}")
    return result


def load_and_merge():
    """メインデータと追加データを統合"""
    print("データ読み込み中...")
    df = pd.read_csv(os.path.join(RAW_DIR, "race_results_big.csv"))
    print(f"  メインデータ: {len(df):,} 行")

    # レーサー成績マージ
    racer_path = os.path.join(EXTRA_DIR, "racer_stats.csv")
    if os.path.exists(racer_path):
        racer_stats = pd.read_csv(racer_path)
        df = df.merge(racer_stats, on="racer_id", how="left")
        print(f"  レーサー成績をマージ")

    # モーター成績マージ
    motor_path = os.path.join(EXTRA_DIR, "motor_stats.csv")
    if os.path.exists(motor_path):
        motor_stats = pd.read_csv(motor_path)
        df = df.merge(motor_stats, on=["jyo_cd", "motor_no"], how="left")
        print(f"  モーター成績をマージ")

    # 場別コース成績マージ
    jyo_path = os.path.join(EXTRA_DIR, "jyo_course_stats.csv")
    if os.path.exists(jyo_path):
        jyo_stats = pd.read_csv(jyo_path)
        df = df.merge(jyo_stats[["jyo_cd", "course", "jyo_course_win_pct", "jyo_course_avg_rank"]],
                       on=["jyo_cd", "course"], how="left")
        print(f"  場別コース成績をマージ")

    # 市場払戻特徴量 (過去90d/180d の会場×コース別平均払戻)
    payout_path = os.path.join(RAW_DIR, "race_payouts.csv")
    if os.path.exists(payout_path):
        payouts = pd.read_csv(payout_path)
        df = add_payout_market_features(df, payouts, windows=(90, 180))
    else:
        print("  警告: race_payouts.csv が見つかりません。市場特徴量をスキップします。")

    return df


def engineer_features(df):
    """特徴量エンジニアリング"""
    print("特徴量生成中...")

    # 日付関連
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # コースのワンホット的な特徴
    df["is_course1"] = (df["course"] == 1).astype(int)
    df["is_inner_course"] = (df["course"] <= 3).astype(int)

    # レーサークラスをエンコード
    class_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    df["racer_class_num"] = df["racer_class"].map(class_map).fillna(0)

    # 天候エンコード
    weather_map = {"晴": 0, "曇り": 1, "雨": 2, "雪": 3}
    df["weather_num"] = df["weather"].map(weather_map).fillna(0)

    # 勝率×コースの交互作用
    df["winrate_x_course"] = df["win_rate"] * (7 - df["course"])

    # ST平均とコースの交互作用
    df["st_x_course"] = df["avg_st"].fillna(0.15) * df["course"]

    # モーター成績×選手成績
    df["motor_racer_score"] = (
        df["motor_top2_pct"].fillna(30) * 0.5 +
        df["win_rate"] * 10
    )

    # ─── レース内相対特徴量 ───────────────────────────────────
    # 同一レース内での各指標の順位・差分を計算
    # これによりモデルが「このレースで誰が一番強いか」を学習できる
    race_group = ["date", "jyo_cd", "race_no"]

    # 勝率の順位 (1=最高)
    df["win_rate_rank"] = df.groupby(race_group)["win_rate"].rank(ascending=False)
    # STの順位 (1=最速)
    df["st_rank"] = df.groupby(race_group)["st"].rank(ascending=True)
    # 展示タイムの順位 (1=最速)
    df["tenji_rank"] = df.groupby(race_group)["tenji_time"].rank(ascending=True)
    # 2連対率の順位 (1=最高)
    df["nirenritsu_rank"] = df.groupby(race_group)["nirenritsu"].rank(ascending=False)

    # レース平均との差分
    df["win_rate_vs_avg"] = df["win_rate"] - df.groupby(race_group)["win_rate"].transform("mean")
    df["st_vs_avg"] = df["st"] - df.groupby(race_group)["st"].transform("mean")
    df["tenji_vs_avg"] = df["tenji_time"] - df.groupby(race_group)["tenji_time"].transform("mean")

    # モーター2連対率のレース内順位
    df["motor_rank"] = df.groupby(race_group)["motor_nirenritsu"].rank(ascending=False)

    # avg_st (集計済み) のレース内順位
    df["avg_st_rank"] = df.groupby(race_group)["avg_st"].rank(ascending=True)
    # ────────────────────────────────────────────────────────

    # ターゲット: 1着かどうか (二値分類)
    df["target_win"] = (df["rank"] == 1).astype(int)

    # ターゲット: 2着かどうか (2連単用)
    df["target_2nd"] = (df["rank"] == 2).astype(int)

    # ターゲット: 3着以内かどうか
    df["target_top3"] = (df["rank"] <= 3).astype(int)

    # 欠損値処理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    print(f"  特徴量数: {len(df.columns)}")
    return df


def split_data(df):
    """時系列を考慮した分割"""
    print("データ分割中...")
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"  Train: {len(train):,} ({train['date'].min()} ~ {train['date'].max()})")
    print(f"  Val:   {len(val):,} ({val['date'].min()} ~ {val['date'].max()})")
    print(f"  Test:  {len(test):,} ({test['date'].min()} ~ {test['date'].max()})")

    return train, val, test


def main():
    print("=" * 60)
    print("ボートレースCSV変換 (特徴量エンジニアリング)")
    print("=" * 60)

    df = load_and_merge()
    df = engineer_features(df)
    train, val, test = split_data(df)

    # 保存
    train.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROC_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)

    # 特徴量リスト保存
    feature_cols = [
        "course", "win_rate", "nirenritsu", "sanrenritsu",
        "motor_nirenritsu", "st", "tenji_time",
        "wind_speed", "wave", "water_temp",
        "month", "dayofweek", "is_weekend",
        "is_course1", "is_inner_course", "racer_class_num", "weather_num",
        "winrate_x_course", "st_x_course", "motor_racer_score",
        "avg_rank", "win_pct", "top2_pct", "top3_pct", "avg_st", "avg_tenji",
        "motor_avg_rank", "motor_win_pct", "motor_top2_pct",
        "jyo_course_win_pct", "jyo_course_avg_rank",
        # レース内相対特徴量
        "win_rate_rank", "st_rank", "tenji_rank", "nirenritsu_rank",
        "win_rate_vs_avg", "st_vs_avg", "tenji_vs_avg",
        "motor_rank", "avg_st_rank",
        # 市場確率特徴量 (過去N日の会場×コース別平均払戻から推定)
        "mkt_avg_payout_90d", "mkt_win_prob_90d",
        "mkt_avg_payout_180d", "mkt_win_prob_180d",
    ]

    with open(os.path.join(PROC_DIR, "feature_cols.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\n使用特徴量 ({len(feature_cols)}個):")
    for c in feature_cols:
        print(f"  - {c}")

    print(f"\n保存先: {PROC_DIR}")
    print("変換完了!")


if __name__ == "__main__":
    main()
