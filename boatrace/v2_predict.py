#!/usr/bin/env python3
"""
v2_predict.py
本番予測: 直近の番組表データを取得し、学習済みモデルで予測する。

公式サイトから番組表 (timetable) を取得できない場合は、
直近の実績データから選手情報を推定して予測を行う。

Usage:
    python v2_predict.py                    # 直近データで予測
    python v2_predict.py --date 2026-03-15  # 指定日で予測
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import requests
import lhafile
import joblib
from io import BytesIO
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v2")

JYO_NAMES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}


def load_models():
    """学習済みモデルとキャリブレーターをロード"""
    models = {}
    for name in ["win", "2nd", "top3"]:
        models[name] = {
            "lgb": lgb.Booster(model_file=os.path.join(CKPT_DIR, f"lgb_{name}.txt")),
            "xgb": xgb.Booster(model_file=os.path.join(CKPT_DIR, f"xgb_{name}.json")),
            "cal": joblib.load(os.path.join(CKPT_DIR, f"cal_{name}.pkl")),
        }
    with open(os.path.join(DATA_DIR, "v2_feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    return models, feature_cols


def predict_calibrated(X, feature_cols, m):
    p_lgb = m["lgb"].predict(X, num_iteration=m["lgb"].best_iteration)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    p_xgb = m["xgb"].predict(dmat, iteration_range=(0, m["xgb"].best_iteration))
    p_ens = (p_lgb + p_xgb) / 2
    logit = np.log(np.clip(p_ens, 1e-6, 1-1e-6) / (1 - np.clip(p_ens, 1e-6, 1-1e-6)))
    return m["cal"].predict_proba(logit.reshape(-1, 1))[:, 1]


def build_racer_history(historical_df):
    """過去データからレーサーごとの統計を構築"""
    df = historical_df.copy()
    df["is_win"] = (df["rank"] == 1).astype(float)
    df["is_top2"] = (df["rank"] <= 2).astype(float)
    df["is_top3"] = (df["rank"] <= 3).astype(float)

    # 全期間統計
    racer_stats = df.groupby("racer_id").agg(
        racer_win_cum_mean=("is_win", "mean"),
        racer_top3_cum_mean=("is_top3", "mean"),
        racer_rank_cum_mean=("rank", "mean"),
        racer_st_cum_mean=("st", "mean"),
        racer_win_cum_count=("is_win", "count"),
    ).reset_index()

    # 直近30レースのEWM統計
    recent = df.sort_values("date").groupby("racer_id").tail(30)
    racer_recent = recent.groupby("racer_id").agg(
        racer_win_ew10=("is_win", lambda x: x.ewm(span=10).mean().iloc[-1] if len(x) > 0 else 0),
        racer_win_ew30=("is_win", lambda x: x.ewm(span=30).mean().iloc[-1] if len(x) > 0 else 0),
        racer_inv_ew10=("rank", lambda x: (1/x).ewm(span=10).mean().iloc[-1] if len(x) > 0 else 0),
        racer_inv_ew30=("rank", lambda x: (1/x).ewm(span=30).mean().iloc[-1] if len(x) > 0 else 0),
    ).reset_index()

    racer_stats = racer_stats.merge(racer_recent, on="racer_id", how="left")

    # コース別勝率
    for c in range(1, 7):
        csub = df[df["course"] == c]
        cstat = csub.groupby("racer_id")["is_win"].mean().reset_index()
        cstat.columns = ["racer_id", f"racer_c{c}_win_rate"]
        racer_stats = racer_stats.merge(cstat, on="racer_id", how="left")

    # 場別勝率
    jyo_stats = df.groupby(["racer_id", "jyo_cd"])["is_win"].mean().reset_index()
    jyo_stats.columns = ["racer_id", "jyo_cd", "racer_jyo_win_cum_mean"]

    # モーター統計
    motor_stats = df.groupby(["jyo_cd", "motor_no"]).agg(
        motor_win_cum_mean=("is_win", "mean"),
        motor_top2_cum_mean=("is_top2", "mean"),
        motor_rank_cum_mean=("rank", "mean"),
        motor_win_cum_count=("is_win", "count"),
    ).reset_index()

    # 場×コース統計
    venue_course = df.groupby(["jyo_cd", "course"])["is_win"].mean().reset_index()
    venue_course.columns = ["jyo_cd", "course", "venue_course_win_cum_mean"]

    return racer_stats, jyo_stats, motor_stats, venue_course


def download_timetable(target_date):
    """番組表 (B ファイル) をダウンロード"""
    ym = target_date.strftime('%Y%m')
    ymd = target_date.strftime('%y%m%d')
    url = f"https://www1.mbrace.or.jp/od2/B/{ym}/b{ymd}.lzh"

    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        lha = lhafile.Lhafile(BytesIO(r.content))
        info = lha.infolist()
        if not info:
            return None
        content = lha.read(info[0].filename)
        return content.decode('cp932', errors='replace')
    except Exception:
        return None


def parse_timetable(text, target_date_str):
    """番組表テキストをパースしてDataFrameを返す"""
    lines = text.splitlines()
    records = []
    i = 0

    while i < len(lines):
        # 会場区切り
        m = re.match(r'(\d{2})BBGN', lines[i])
        if m:
            jyo_cd = m.group(1)
            end_marker = f'{jyo_cd}BEND'
            i += 1

            current_race = None
            while i < len(lines) and lines[i] != end_marker:
                line = lines[i]

                # レースヘッダー: "   1R  ..." or "  1Ｒ ..."
                rm = re.search(r'(\d+)[RＲ]', line)
                if rm and ('Ｈ' in line or 'H' in line):
                    current_race = int(rm.group(1))
                    # 天候情報もここから取得
                    weather = None
                    for w in ['晴', '曇', '雨', '雪']:
                        if w in line:
                            weather = '曇り' if w == '曇' else w
                            break
                    wind_speed = None
                    wm = re.search(r'(\d+)m', line)
                    if wm:
                        wind_speed = int(wm.group(1))
                    wave = None
                    wvm = re.search(r'(\d+)cm', line)
                    if wvm:
                        wave = int(wvm.group(1))
                    i += 1
                    continue

                # 番組行: "1 3527 中嶋　誠一郎 28 ..."
                if current_race and re.match(r'\d\s+\d{4}', line.strip()):
                    parts = line.strip()
                    try:
                        course = int(parts[0])
                        racer_id = int(parts[2:6])
                        # 他のフィールドは位置依存で取得
                        records.append({
                            "date": target_date_str,
                            "jyo_cd": jyo_cd,
                            "jyo_name": JYO_NAMES.get(jyo_cd, "不明"),
                            "race_no": current_race,
                            "course": course,
                            "racer_id": racer_id,
                            "weather": weather,
                            "wind_speed": wind_speed,
                            "wave": wave,
                        })
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue
        i += 1

    return pd.DataFrame(records) if records else None


def prepare_features(df, racer_stats, jyo_stats, motor_stats, venue_course, feature_cols):
    """予測用特徴量を構築"""
    df = df.copy()
    df["jyo_cd"] = df["jyo_cd"].astype(str).str.zfill(2)

    # レーサー統計マージ
    df = df.merge(racer_stats, on="racer_id", how="left")

    # 場別レーサー統計
    df = df.merge(jyo_stats, on=["racer_id", "jyo_cd"], how="left")

    # コース別勝率
    df["racer_course_win_rate"] = np.nan
    for c in range(1, 7):
        col = f"racer_c{c}_win_rate"
        if col in df.columns:
            mask = df["course"] == c
            df.loc[mask, "racer_course_win_rate"] = df.loc[mask, col]

    # モーター統計
    if "motor_no" in df.columns:
        df = df.merge(motor_stats, on=["jyo_cd", "motor_no"], how="left")

    # 場×コース統計
    df = df.merge(venue_course, on=["jyo_cd", "course"], how="left")

    # 展示タイム・ST (番組表には含まれないため推定値)
    if "st" not in df.columns or df["st"].isna().all():
        df["st"] = df["racer_st_cum_mean"].fillna(0.15)
    if "tenji_time" not in df.columns or df["tenji_time"].isna().all():
        df["tenji_time"] = 6.75  # デフォルト値

    # レース内相対特徴量
    race_key = ["date", "jyo_cd", "race_no"]
    for col, asc, new in [
        ("racer_win_cum_mean", False, "rr_win_rank"),
        ("racer_st_cum_mean", True, "rr_st_rank"),
        ("racer_top3_cum_mean", False, "rr_top3_rank"),
        ("motor_win_cum_mean", False, "rr_motor_rank"),
        ("tenji_time", True, "rr_tenji_rank"),
    ]:
        if col in df.columns:
            df[new] = df.groupby(race_key)[col].rank(ascending=asc, method="min")

    for col, new in [
        ("racer_win_cum_mean", "rr_win_vs_avg"),
        ("racer_st_cum_mean", "rr_st_vs_avg"),
        ("tenji_time", "rr_tenji_vs_avg"),
    ]:
        if col in df.columns:
            df[new] = df[col] - df.groupby(race_key)[col].transform("mean")

    if "racer_win_cum_mean" in df.columns:
        df["rr_win_vs_best"] = (
            df["racer_win_cum_mean"]
            - df.groupby(race_key)["racer_win_cum_mean"].transform("max")
        )

    df["rr_n_boats"] = df.groupby(race_key)["course"].transform("count")

    # 交互作用
    win_mean = df.get("racer_win_cum_mean", pd.Series(0.167, index=df.index)).fillna(0.167)
    df["ix_course_x_racer"] = (7 - df["course"]) / 6.0 * win_mean
    motor_mean = df.get("motor_win_cum_mean", pd.Series(0.167, index=df.index)).fillna(0.167)
    df["ix_motor_x_racer"] = motor_mean * win_mean
    st_mean = df.get("racer_st_cum_mean", pd.Series(0.15, index=df.index)).fillna(0.15)
    df["ix_st_x_course"] = st_mean * df["course"]
    df["ix_tenji_x_racer"] = df["tenji_time"].fillna(6.75) * win_mean

    # カレンダー
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 天候
    weather_map = {"晴": 0, "曇り": 1, "雨": 2, "雪": 3}
    if "weather" in df.columns:
        df["weather_num"] = df["weather"].map(weather_map).fillna(0)
    else:
        df["weather_num"] = 0
    if "wind_speed" not in df.columns:
        df["wind_speed"] = 3
    if "wave" not in df.columns:
        df["wave"] = 3

    # 欠損値補完
    for c in feature_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
        elif c not in df.columns:
            df[c] = 0

    return df


def display_predictions(exacta_df):
    """予測結果を表示"""
    print(f"\n{'═' * 80}")
    print("2連単 予測結果")
    print(f"{'═' * 80}")

    for jyo_cd, jyo_group in exacta_df.groupby("jyo_cd"):
        jyo_name = JYO_NAMES.get(jyo_cd, jyo_cd)
        print(f"\n{'━' * 78}")
        print(f"  {jyo_name} ({jyo_cd})")
        print(f"{'━' * 78}")
        print(f"  {'R':>3}  {'1着→2着':>8}  {'1着確率':>7}  {'2連単確率':>9}  {'損益分岐':>8}  {'判定'}")

        for _, row in jyo_group.sort_values("race_no").iterrows():
            be = int(100 / max(row["p_exacta"], 0.001))
            be = max(100, (be // 100) * 100)
            # 確信度ランク
            if row["p_exacta"] >= 0.08:
                grade = "◎"
            elif row["p_exacta"] >= 0.05:
                grade = "○"
            elif row["p_exacta"] >= 0.03:
                grade = "△"
            else:
                grade = "▽"

            print(f"  {int(row['race_no']):>3}R  "
                  f"{int(row['pred_1st'])}-{int(row['pred_2nd'])}      "
                  f"{row['p_win']:>6.1%}  {row['p_exacta']:>8.1%}  "
                  f"{be:>7,}円  {grade}")

    # 注目レース (確信度Top10)
    print(f"\n{'═' * 78}")
    print("注目レース (モデル確信度 Top 10)")
    print(f"{'═' * 78}")
    top = exacta_df.sort_values("p_exacta", ascending=False).head(10)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        jyo_name = JYO_NAMES.get(row["jyo_cd"], row["jyo_cd"])
        be = int(100 / max(row["p_exacta"], 0.001))
        be = max(100, (be // 100) * 100)
        print(f"  {i:>2}. {jyo_name:>4} {int(row['race_no']):>2}R  "
              f"【{int(row['pred_1st'])}-{int(row['pred_2nd'])}】  "
              f"2連単確率: {row['p_exacta']:.1%}  "
              f"損益分岐: {be:,}円")

    print(f"\n※ 実際のオッズが「損益分岐」を超えていれば期待値プラス")


def main():
    parser = argparse.ArgumentParser(description='v2 予測')
    parser.add_argument('--date', default=None, help='予測対象日 (YYYY-MM-DD)')
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        target_date = datetime.now()

    target_str = target_date.strftime('%Y-%m-%d')
    print("=" * 60)
    print(f"v2 ボートレース予測 - {target_str}")
    print("=" * 60)

    # モデルロード
    print("モデル読み込み中...")
    models, feature_cols = load_models()

    # 過去データからレーサー統計を構築
    print("レーサー履歴構築中...")
    hist_path = os.path.join(RAW_DIR, "race_results_real.csv")
    hist_df = pd.read_csv(hist_path, dtype={"jyo_cd": str})
    hist_df["jyo_cd"] = hist_df["jyo_cd"].str.zfill(2)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    # 予測日より前のデータのみ使用
    hist_df = hist_df[hist_df["date"] < target_str]
    print(f"  履歴データ: {len(hist_df):,} 行")

    racer_stats, jyo_stats, motor_stats, venue_course = build_racer_history(hist_df)
    print(f"  レーサー数: {len(racer_stats):,}")

    # 番組表取得を試みる
    print(f"\n番組表取得中 ({target_str})...")
    timetable_text = download_timetable(target_date)

    if timetable_text:
        race_df = parse_timetable(timetable_text, target_str)
        if race_df is not None and len(race_df) > 0:
            print(f"  番組表から {len(race_df)} エントリー取得")
        else:
            print("  番組表のパースに失敗。直近の結果データから推定します。")
            timetable_text = None
    else:
        print("  番組表が取得できません。直近の結果データから推定します。")

    if timetable_text is None or race_df is None or len(race_df) == 0:
        # 直近の結果データから当日のレース情報を推定
        # 直近3日のデータを使用
        recent = hist_df.tail(hist_df.groupby(["date"]).ngroups * 6 * 12).copy()
        latest_date = recent["date"].max()
        race_df = recent[recent["date"] == latest_date].copy()
        race_df["date"] = target_str
        print(f"  最新日 ({latest_date.date()}) のレース構成を使用: {len(race_df)} エントリー")

    # 特徴量準備
    print("\n特徴量生成中...")
    df = prepare_features(race_df, racer_stats, jyo_stats, motor_stats, venue_course, feature_cols)

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values

    # 予測
    print("予測実行中...")
    df["p_win"] = predict_calibrated(X, available, models["win"])
    df["p_2nd"] = predict_calibrated(X, available, models["2nd"])

    # 2連単予測
    exacta_rows = []
    for (jyo_cd, race_no), g in df.groupby(["jyo_cd", "race_no"]):
        idx1 = g["p_win"].idxmax()
        c1 = g.loc[idx1, "course"]
        p1 = g.loc[idx1, "p_win"]

        rest = g[g.index != idx1]
        if len(rest) == 0:
            continue
        p2_sum = rest["p_2nd"].sum()
        idx2 = rest["p_2nd"].idxmax()
        c2 = rest.loc[idx2, "course"]
        p2 = rest.loc[idx2, "p_2nd"]
        p2_cond = p2 / p2_sum if p2_sum > 0 else p2

        exacta_rows.append({
            "jyo_cd": jyo_cd, "race_no": race_no,
            "pred_1st": int(c1), "pred_2nd": int(c2),
            "p_win": p1, "p_2nd_cond": p2_cond,
            "p_exacta": p1 * p2_cond,
        })

    exacta_df = pd.DataFrame(exacta_rows)

    # 表示
    display_predictions(exacta_df)

    # CSV保存
    out_path = os.path.join(CKPT_DIR, f"v2_predictions_{target_date.strftime('%Y%m%d')}.csv")
    exacta_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n予測結果CSV: {out_path}")
    print("完了!")


if __name__ == "__main__":
    main()
