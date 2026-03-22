#!/usr/bin/env python3
"""
generate_payout_stats.py
race_payouts.csv から「場別×コース別 市場払戻特徴量」の静的ルックアップテーブルを生成する。

出力: data/extra/jyo_course_payout_stats.csv
  列: jyo_cd, course, mkt_avg_payout_90d, mkt_win_prob_90d,
               mkt_avg_payout_180d, mkt_win_prob_180d, n_90d, n_180d

■ 使い方
  python generate_payout_stats.py          # race_payouts.csv から生成
  python generate_payout_stats.py --days 360   # 参照期間を拡大 (デフォルト: 180d)

■ convert_boatracecsv.py との違い
  学習時: rolling(closed='left') で当日のレースを除外 (データリーク防止)
  予測時: 本スクリプトが生成するスナップショット = 最新データまで含めた平均
  → 当日レースの予測に使う場合は前日までのデータが入るため問題なし
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta

RAW_DIR   = os.path.join(os.path.dirname(__file__), "data", "raw")
EXTRA_DIR = os.path.join(os.path.dirname(__file__), "data", "extra")
os.makedirs(EXTRA_DIR, exist_ok=True)


def generate(payout_path: str,
             windows: tuple[int, ...] = (90, 180),
             min_periods: int = 5) -> pd.DataFrame:
    """
    race_payouts.csv を読み込んで (jyo_cd, course) ごとの
    直近 N 日平均払戻とその市場暗示確率を計算する。

    Parameters
    ----------
    payout_path  : race_payouts.csv のパス
    windows      : 計算する日数ウィンドウ (デフォルト: 90日・180日)
    min_periods  : 最低サンプル数 (これ未満は NaN にする)
    """
    print(f"race_payouts.csv 読み込み中: {payout_path}")
    df = pd.read_csv(payout_path, dtype={"jyo_cd": str})
    df["jyo_cd"] = df["jyo_cd"].str.zfill(2)
    df["date"]   = pd.to_datetime(df["date"])
    df = df.rename(columns={"rank1_course": "course"})

    latest_date = df["date"].max()
    print(f"  データ期間: {df['date'].min().date()} ~ {latest_date.date()}")
    print(f"  総レース数: {len(df):,}  会場数: {df['jyo_cd'].nunique()}")

    rows = []
    for (jyo_cd, course), grp in df.groupby(["jyo_cd", "course"]):
        grp = grp.sort_values("date")
        record = {"jyo_cd": jyo_cd, "course": int(course)}

        for w in windows:
            cutoff = latest_date - timedelta(days=w)
            window_data = grp[grp["date"] >= cutoff]["payout_2nd"]
            n = len(window_data)
            record[f"n_{w}d"] = n

            if n >= min_periods:
                avg = window_data.mean()
                record[f"mkt_avg_payout_{w}d"] = round(avg, 1)
                # 市場暗示確率: 100 / avg_payout (払戻最低 200 円でクリップ)
                record[f"mkt_win_prob_{w}d"] = round(100.0 / max(avg, 200), 6)
            else:
                record[f"mkt_avg_payout_{w}d"] = np.nan
                record[f"mkt_win_prob_{w}d"]   = np.nan

        rows.append(record)

    result = pd.DataFrame(rows)

    # 列を整理
    base_cols = ["jyo_cd", "course"]
    feat_cols = []
    for w in windows:
        feat_cols += [f"mkt_avg_payout_{w}d", f"mkt_win_prob_{w}d", f"n_{w}d"]
    result = result[base_cols + feat_cols]

    return result


def main():
    parser = argparse.ArgumentParser(description="場別コース別市場払戻統計を生成")
    parser.add_argument("--input",  default=os.path.join(RAW_DIR, "race_payouts.csv"),
                        help="入力ファイル (デフォルト: data/raw/race_payouts.csv)")
    parser.add_argument("--output", default=os.path.join(EXTRA_DIR, "jyo_course_payout_stats.csv"),
                        help="出力ファイル (デフォルト: data/extra/jyo_course_payout_stats.csv)")
    parser.add_argument("--days",   type=int, default=180,
                        help="最大ウィンドウ日数 (デフォルト: 180)")
    parser.add_argument("--min-periods", type=int, default=5,
                        help="最低サンプル数 (デフォルト: 5)")
    args = parser.parse_args()

    # ウィンドウ: 最大 N 日と半分 (例: 90d/180d)
    max_w = args.days
    half_w = max_w // 2
    windows = (half_w, max_w)

    print("=" * 60)
    print("場別コース別 市場払戻統計 生成")
    print("=" * 60)
    print(f"ウィンドウ: {windows[0]}日 / {windows[1]}日")
    print(f"最低サンプル数: {args.min_periods}")

    if not os.path.exists(args.input):
        print(f"エラー: {args.input} が見つかりません。")
        return

    result = generate(args.input, windows=windows, min_periods=args.min_periods)

    # ── サマリー ────────────────────────────────────────────────────────────
    w_short, w_long = windows
    n_total   = len(result)
    n_short   = result[f"mkt_avg_payout_{w_short}d"].notna().sum()
    n_long    = result[f"mkt_avg_payout_{w_long}d"].notna().sum()

    print(f"\n生成結果:")
    print(f"  (jyo_cd, course) ペア数: {n_total}")
    print(f"  {w_short}日窓 有効: {n_short}/{n_total}  "
          f"({n_short/n_total:.1%})  NaN: {n_total-n_short}")
    print(f"  {w_long}日窓 有効: {n_long}/{n_total}  "
          f"({n_long/n_total:.1%})  NaN: {n_total-n_long}")

    # コース別の統計を表示
    print(f"\n  コース別 平均払戻 ({w_long}日窓, 全会場平均):")
    course_summary = (result.groupby("course")[f"mkt_avg_payout_{w_long}d"]
                      .agg(["mean", "std", "count"])
                      .round(1))
    print(f"  {'コース':>6} {'平均払戻':>9} {'標準偏差':>9} {'会場数':>6} {'市場確率':>9}")
    print(f"  {'─' * 45}")
    for course, row in course_summary.iterrows():
        avg = row["mean"]
        prob = 100.0 / max(avg, 200) if pd.notna(avg) else float("nan")
        print(f"  {'コース'+str(course):>6} {avg:>9.0f}円 {row['std']:>9.0f}  "
              f"{int(row['count']):>6}  {prob:>8.1%}")

    # ── 保存 ────────────────────────────────────────────────────────────────
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n保存: {args.output}")
    print("完了!")


if __name__ == "__main__":
    main()
