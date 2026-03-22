#!/usr/bin/env python3
"""
download_extra.py
追加の特徴量データを生成する:
- レーサー直近成績
- モーター成績
- 場別成績
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EXTRA_DIR = os.path.join(DATA_DIR, "extra")
os.makedirs(EXTRA_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("追加データ生成 (レーサー成績・モーター成績・場別傾向)")
    print("=" * 60)

    # メインデータ読み込み
    big_path = os.path.join(RAW_DIR, "race_results_big.csv")
    if not os.path.exists(big_path):
        print("エラー: race_results_big.csv が見つかりません。先に download_big.py を実行してください。")
        return

    df = pd.read_csv(big_path)
    print(f"メインデータ: {len(df):,} 行")

    # 1. レーサー別集計
    print("レーサー別成績を集計中...")
    racer_stats = df.groupby("racer_id").agg(
        total_races=("rank", "count"),
        avg_rank=("rank", "mean"),
        win_count=("rank", lambda x: (x == 1).sum()),
        top2_count=("rank", lambda x: (x <= 2).sum()),
        top3_count=("rank", lambda x: (x <= 3).sum()),
        avg_st=("st", "mean"),
        avg_tenji=("tenji_time", "mean"),
    ).reset_index()

    racer_stats["win_pct"] = (racer_stats["win_count"] / racer_stats["total_races"] * 100).round(1)
    racer_stats["top2_pct"] = (racer_stats["top2_count"] / racer_stats["total_races"] * 100).round(1)
    racer_stats["top3_pct"] = (racer_stats["top3_count"] / racer_stats["total_races"] * 100).round(1)
    racer_stats["avg_rank"] = racer_stats["avg_rank"].round(2)
    racer_stats["avg_st"] = racer_stats["avg_st"].round(3)
    racer_stats["avg_tenji"] = racer_stats["avg_tenji"].round(3)

    racer_path = os.path.join(EXTRA_DIR, "racer_stats.csv")
    racer_stats.to_csv(racer_path, index=False, encoding="utf-8-sig")
    print(f"  レーサー数: {len(racer_stats):,}")
    print(f"  保存先: {racer_path}")

    # 2. モーター別集計
    print("モーター別成績を集計中...")
    motor_stats = df.groupby(["jyo_cd", "motor_no"]).agg(
        motor_races=("rank", "count"),
        motor_avg_rank=("rank", "mean"),
        motor_win_count=("rank", lambda x: (x == 1).sum()),
        motor_top2_count=("rank", lambda x: (x <= 2).sum()),
    ).reset_index()

    motor_stats["motor_win_pct"] = (motor_stats["motor_win_count"] / motor_stats["motor_races"] * 100).round(1)
    motor_stats["motor_top2_pct"] = (motor_stats["motor_top2_count"] / motor_stats["motor_races"] * 100).round(1)
    motor_stats["motor_avg_rank"] = motor_stats["motor_avg_rank"].round(2)

    motor_path = os.path.join(EXTRA_DIR, "motor_stats.csv")
    motor_stats.to_csv(motor_path, index=False, encoding="utf-8-sig")
    print(f"  モーター数: {len(motor_stats):,}")
    print(f"  保存先: {motor_path}")

    # 3. 場別・コース別集計
    print("場別・コース別成績を集計中...")
    jyo_course_stats = df.groupby(["jyo_cd", "jyo_name", "course"]).agg(
        jyo_races=("rank", "count"),
        jyo_course_avg_rank=("rank", "mean"),
        jyo_course_win_count=("rank", lambda x: (x == 1).sum()),
    ).reset_index()

    jyo_course_stats["jyo_course_win_pct"] = (
        jyo_course_stats["jyo_course_win_count"] / jyo_course_stats["jyo_races"] * 100
    ).round(1)
    jyo_course_stats["jyo_course_avg_rank"] = jyo_course_stats["jyo_course_avg_rank"].round(2)

    jyo_path = os.path.join(EXTRA_DIR, "jyo_course_stats.csv")
    jyo_course_stats.to_csv(jyo_path, index=False, encoding="utf-8-sig")
    print(f"  場×コース組合せ: {len(jyo_course_stats):,}")
    print(f"  保存先: {jyo_path}")

    # 4. 天候別集計
    print("天候別成績を集計中...")
    weather_stats = df.groupby(["weather", "course"]).agg(
        w_races=("rank", "count"),
        w_avg_rank=("rank", "mean"),
        w_win_count=("rank", lambda x: (x == 1).sum()),
    ).reset_index()

    weather_stats["w_course_win_pct"] = (
        weather_stats["w_win_count"] / weather_stats["w_races"] * 100
    ).round(1)

    weather_path = os.path.join(EXTRA_DIR, "weather_stats.csv")
    weather_stats.to_csv(weather_path, index=False, encoding="utf-8-sig")
    print(f"  保存先: {weather_path}")

    # 5. 直近フォーム統計 (予測時マージ用スナップショット)
    print("直近フォーム統計を集計中...")
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.sort_values(["racer_id", "date"]).reset_index(drop=True)
    df2["win"]  = (df2["rank"] == 1).astype(float)
    df2["top2"] = (df2["rank"] <= 2).astype(float)

    form_records = []
    for racer_id, gdf in df2.groupby("racer_id"):
        wins  = gdf["win"].values
        top2s = gdf["top2"].values
        ranks = gdf["rank"].values
        sts   = gdf["st"].values
        row = {"racer_id": racer_id}
        for n in (3, 5, 10):
            sl = slice(max(0, len(gdf) - n), len(gdf))
            row[f"form_win_{n}"]      = float(np.mean(wins[sl]))  if len(gdf) >= 1 else np.nan
            row[f"form_top2_{n}"]     = float(np.mean(top2s[sl])) if len(gdf) >= 1 else np.nan
            row[f"form_avg_rank_{n}"] = float(np.mean(ranks[sl])) if len(gdf) >= 1 else np.nan
            row[f"form_avg_st_{n}"]   = float(np.nanmean(sts[sl])) if len(gdf) >= 1 else np.nan
        sl30 = slice(max(0, len(gdf) - 30), len(gdf))
        sl5  = slice(max(0, len(gdf) - 5),  len(gdf))
        sl3  = slice(max(0, len(gdf) - 3),  len(gdf))
        sl10 = slice(max(0, len(gdf) - 10), len(gdf))
        row["form_trend_win"]  = float(np.mean(wins[sl5]))  - float(np.mean(wins[sl30]))
        row["form_trend_rank"] = float(np.mean(ranks[sl3])) - float(np.mean(ranks[sl10]))
        ws = 0
        for w in reversed(wins.tolist()):
            if w == 1: ws += 1
            else: break
        ls = 0
        for w in reversed(wins.tolist()):
            if w == 0: ls += 1
            else: break
        row["form_win_streak"]    = ws
        row["form_no_win_streak"] = ls
        form_records.append(row)

    form_df = pd.DataFrame(form_records)
    form_path = os.path.join(EXTRA_DIR, "racer_recent_form.csv")
    form_df.to_csv(form_path, index=False, encoding="utf-8-sig")
    print(f"  レーサー数: {len(form_df):,}")
    print(f"  保存先: {form_path}")

    # 6. 同場直近フォーム (racer_id × jyo_cd)
    print("同場直近フォーム統計を集計中...")
    venue_records = []
    for (racer_id, jyo_cd), gdf in df2.groupby(["racer_id", "jyo_cd"]):
        wins  = gdf["win"].values
        top2s = gdf["top2"].values
        sl5 = slice(max(0, len(gdf) - 5), len(gdf))
        venue_records.append({
            "racer_id":         racer_id,
            "jyo_cd":           str(jyo_cd).zfill(2),
            "form_venue_win_5":  float(np.mean(wins[sl5])),
            "form_venue_top2_5": float(np.mean(top2s[sl5])),
        })
    venue_df = pd.DataFrame(venue_records)
    venue_path = os.path.join(EXTRA_DIR, "racer_venue_form.csv")
    venue_df.to_csv(venue_path, index=False, encoding="utf-8-sig")
    print(f"  racer×場 組合せ: {len(venue_df):,}")
    print(f"  保存先: {venue_path}")

    print("\n追加データ生成 完了!")


if __name__ == "__main__":
    main()
