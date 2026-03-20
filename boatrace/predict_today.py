#!/usr/bin/env python3
"""
predict_today.py
今日のレースに対して予測を行う。
学習済みモデルを読み込み、本日のレース情報を生成して予測する。
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "real55k")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
EXTRA_DIR = os.path.join(os.path.dirname(__file__), "data", "extra")


def generate_today_races():
    """今日のレースデータを生成 (デモ用)"""
    np.random.seed(int(datetime.now().strftime("%Y%m%d")))

    today = datetime.now().strftime("%Y-%m-%d")

    jyo_info = [
        ("06", "浜名湖"), ("12", "住之江"), ("18", "徳山"),
        ("24", "大村"), ("03", "江戸川"), ("15", "丸亀"),
    ]

    records = []
    for jyo_cd, jyo_name in jyo_info:
        n_races = np.random.randint(10, 13)
        for race_no in range(1, n_races + 1):
            weather = np.random.choice(["晴", "曇り", "雨"], p=[0.5, 0.35, 0.15])
            wind_speed = max(0, int(np.random.normal(3, 2)))
            wave = max(1, int(np.random.normal(4, 2)))
            water_temp = round(np.random.normal(14, 2), 1)  # 3月

            for course in range(1, 7):
                racer_id = np.random.randint(2000, 5000)
                racer_class = np.random.choice(["A1", "A2", "B1", "B2"], p=[0.15, 0.20, 0.45, 0.20])

                class_rates = {"A1": 7.2, "A2": 5.8, "B1": 4.5, "B2": 3.2}
                win_rate = round(np.random.normal(class_rates[racer_class], 0.5), 2)
                win_rate = max(1.0, min(9.5, win_rate))

                nirenritsu = round(min(90, max(5, win_rate * 8 + np.random.normal(0, 5))), 1)
                sanrenritsu = round(min(95, max(10, nirenritsu + np.random.normal(8, 3))), 1)

                motor_no = np.random.randint(1, 80)
                boat_no = np.random.randint(1, 80)
                motor_nirenritsu = round(max(15, min(70, np.random.normal(35, 8))), 1)
                st = round(max(0.01, np.random.normal(0.15, 0.04)), 2)
                tenji_time = round(np.random.normal(6.75, 0.1), 2)

                records.append({
                    "date": today,
                    "jyo_cd": jyo_cd,
                    "jyo_name": jyo_name,
                    "race_no": race_no,
                    "course": course,
                    "racer_id": racer_id,
                    "racer_class": racer_class,
                    "motor_no": motor_no,
                    "boat_no": boat_no,
                    "win_rate": win_rate,
                    "nirenritsu": nirenritsu,
                    "sanrenritsu": sanrenritsu,
                    "motor_nirenritsu": motor_nirenritsu,
                    "st": st,
                    "tenji_time": tenji_time,
                    "weather": weather,
                    "wind_speed": wind_speed,
                    "wave": wave,
                    "water_temp": water_temp,
                })

    return pd.DataFrame(records)


def prepare_features(df):
    """予測用の特徴量を準備"""
    df = df.copy()

    # 日付特徴量
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["is_course1"] = (df["course"] == 1).astype(int)
    df["is_inner_course"] = (df["course"] <= 3).astype(int)

    class_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    df["racer_class_num"] = df["racer_class"].map(class_map).fillna(0)

    weather_map = {"晴": 0, "曇り": 1, "雨": 2, "雪": 3}
    df["weather_num"] = df["weather"].map(weather_map).fillna(0)

    df["winrate_x_course"] = df["win_rate"] * (7 - df["course"])
    df["st_x_course"] = df["st"] * df["course"]
    df["motor_racer_score"] = df["motor_nirenritsu"] * 0.5 + df["win_rate"] * 10

    # 追加統計がある場合はマージ
    racer_path = os.path.join(EXTRA_DIR, "racer_stats.csv")

    if os.path.exists(racer_path):
        racer_stats = pd.read_csv(racer_path)
        df = df.merge(racer_stats, on="racer_id", how="left")

    motor_path = os.path.join(EXTRA_DIR, "motor_stats.csv")
    if os.path.exists(motor_path):
        motor_stats = pd.read_csv(motor_path)
        motor_stats["jyo_cd"] = motor_stats["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(motor_stats, on=["jyo_cd", "motor_no"], how="left")

    jyo_path = os.path.join(EXTRA_DIR, "jyo_course_stats.csv")
    if os.path.exists(jyo_path):
        jyo_stats = pd.read_csv(jyo_path)
        jyo_stats["jyo_cd"] = jyo_stats["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(
            jyo_stats[["jyo_cd", "course", "jyo_course_win_pct", "jyo_course_avg_rank"]],
            on=["jyo_cd", "course"], how="left"
        )

    # ─── レース内相対特徴量 (学習時と同じ特徴量を生成) ────────
    race_group = ["date", "jyo_cd", "race_no"]
    df["win_rate_rank"] = df.groupby(race_group)["win_rate"].rank(ascending=False)
    df["st_rank"] = df.groupby(race_group)["st"].rank(ascending=True)
    df["tenji_rank"] = df.groupby(race_group)["tenji_time"].rank(ascending=True)
    df["nirenritsu_rank"] = df.groupby(race_group)["nirenritsu"].rank(ascending=False)
    df["win_rate_vs_avg"] = df["win_rate"] - df.groupby(race_group)["win_rate"].transform("mean")
    df["st_vs_avg"] = df["st"] - df.groupby(race_group)["st"].transform("mean")
    df["tenji_vs_avg"] = df["tenji_time"] - df.groupby(race_group)["tenji_time"].transform("mean")
    df["motor_rank"] = df.groupby(race_group)["motor_nirenritsu"].rank(ascending=False)
    avg_st_col = "avg_st" if "avg_st" in df.columns else "st"
    df["avg_st_rank"] = df.groupby(race_group)[avg_st_col].rank(ascending=True)
    # ────────────────────────────────────────────────────────

    # 欠損値処理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def main():
    print("=" * 60)
    print(f"ボートレース予測 - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)

    # モデル読み込み
    win_model_path = os.path.join(CKPT_DIR, "lgbm_win.txt")
    snd_model_path = os.path.join(CKPT_DIR, "lgbm_2nd.txt")

    if not os.path.exists(win_model_path) or not os.path.exists(snd_model_path):
        print("エラー: モデルが見つかりません。先に run_train55k.py を実行してください。")
        return

    print("モデル読み込み中...")
    model_win = lgb.Booster(model_file=win_model_path)
    model_2nd = lgb.Booster(model_file=snd_model_path)

    # 特徴量リスト
    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # 今日のレースデータ
    print("本日のレースデータを取得中...")
    df = generate_today_races()
    df = prepare_features(df)

    print(f"対象レース数: {df['race_no'].nunique()} レース × {df['jyo_cd'].nunique()} 場")
    print(f"対象エントリー: {len(df)} 艇")

    # 予測
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].values

    df["prob_win"] = model_win.predict(X)
    df["prob_2nd"] = model_2nd.predict(X)

    # 2連単予測: レースごとに1着・2着を確定
    exacta_rows = []
    for (jyo_cd, jyo_name, race_no), race_df in df.groupby(["jyo_cd", "jyo_name", "race_no"]):
        race_df = race_df.copy()
        # 1着予測: prob_win 最大
        pred_1st_idx = race_df["prob_win"].idxmax()
        pred_1st = race_df.loc[pred_1st_idx]
        # 2着予測: 1着を除いて prob_2nd 最大
        rest = race_df[race_df.index != pred_1st_idx]
        pred_2nd_idx = rest["prob_2nd"].idxmax()
        pred_2nd = rest.loc[pred_2nd_idx]

        # 信頼度スコア: 1着確率 × 2着確率
        confidence_score = pred_1st["prob_win"] * pred_2nd["prob_2nd"]

        exacta_rows.append({
            "jyo_cd": jyo_cd,
            "jyo_name": jyo_name,
            "race_no": race_no,
            "course_1st": int(pred_1st["course"]),
            "class_1st": pred_1st["racer_class"],
            "prob_win": pred_1st["prob_win"],
            "course_2nd": int(pred_2nd["course"]),
            "class_2nd": pred_2nd["racer_class"],
            "prob_2nd": pred_2nd["prob_2nd"],
            "confidence_score": confidence_score,
        })

    exacta_df = pd.DataFrame(exacta_rows)

    # 結果表示
    print("\n" + "=" * 80)
    print("2連単 予測結果")
    print("=" * 80)

    for (jyo_cd, jyo_name), jyo_df in exacta_df.groupby(["jyo_cd", "jyo_name"]):
        print(f"\n{'━' * 70}")
        print(f"  {jyo_name} ({jyo_cd})")
        print(f"{'━' * 70}")
        print(f"  {'R':>3}  {'1着→2着':^16}  {'1着確率':>8}  {'2着確率':>8}  信頼")

        for _, row in jyo_df.sort_values("race_no").iterrows():
            confidence = "★★★" if row["confidence_score"] > 0.08 else "★★" if row["confidence_score"] > 0.04 else "★"
            print(f"  {int(row['race_no']):2d}R  "
                  f"{row['course_1st']}コース({row['class_1st']})→{row['course_2nd']}コース({row['class_2nd']})  "
                  f"{row['prob_win']:>7.1%}  {row['prob_2nd']:>7.1%}  {confidence}")

    # 高信頼レースの抽出
    print(f"\n{'=' * 70}")
    print("高信頼 2連単 (信頼スコア上位)")
    print(f"{'=' * 70}")
    top_exacta = exacta_df.sort_values("confidence_score", ascending=False).head(10)
    for _, row in top_exacta.iterrows():
        print(f"  {row['jyo_name']} {int(row['race_no']):2d}R  "
              f"【{row['course_1st']}-{row['course_2nd']}】  "
              f"1着{row['prob_win']:.1%} × 2着{row['prob_2nd']:.1%}")

    # 予測結果CSV保存 (2連単形式)
    output_path = os.path.join(CKPT_DIR, f"predictions_{datetime.now().strftime('%Y%m%d')}.csv")
    exacta_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n予測結果CSV: {output_path}")

    print("\n予測完了!")


if __name__ == "__main__":
    main()
