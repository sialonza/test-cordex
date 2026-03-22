#!/usr/bin/env python3
"""
backtest_ev.py
テストデータを使って2連単の期待値 (EV) バックテストを行う。

出力:
  - EV閾値別の的中率・ROI・推奨ベット数
  - 資金曲線 (プリント)
  - backtest_result.csv
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints", "real55k")

TAKE_RATE  = 0.225   # 2連単控除率


def load_models():
    model_win = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_win.txt"))
    model_2nd = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_2nd.txt"))
    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    thr_path = os.path.join(CKPT_DIR, "thresholds.json")
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            thresholds = json.load(f)
        print(f"最適閾値を読み込み: {thresholds}")
    else:
        thresholds = {"lgbm_win": 0.5, "lgbm_2nd": 0.5}

    return model_win, model_2nd, feature_cols, thresholds


def predict_exacta(test_df, model_win, model_2nd, feature_cols, thresholds=None):
    """テストデータ全レースの2連単予測を返す DataFrame。"""
    if thresholds is None:
        thresholds = {"lgbm_win": 0.5, "lgbm_2nd": 0.5}
    thr_win = thresholds.get("lgbm_win", 0.5)
    thr_2nd = thresholds.get("lgbm_2nd", 0.5)

    available = [c for c in feature_cols if c in test_df.columns]
    X = test_df[available].values

    test_df = test_df.copy()
    test_df["prob_win"] = model_win.predict(X, num_iteration=model_win.best_iteration)
    test_df["prob_2nd"] = model_2nd.predict(X, num_iteration=model_2nd.best_iteration)

    rows = []
    for (date, jyo_cd, race_no), g in test_df.groupby(["date", "jyo_cd", "race_no"]):
        # 1着予測: prob_win が閾値超えの中で最大 (なければ確率最大)
        win_cands = g[g["prob_win"] >= thr_win]
        if len(win_cands) == 0:
            win_cands = g
        idx1 = win_cands["prob_win"].idxmax()
        c1   = g.loc[idx1, "course"]
        p1   = g.loc[idx1, "prob_win"]
        # 2着予測: 1着除外後 prob_2nd が閾値超えの中で最大
        rest = g[g.index != idx1]
        if len(rest) == 0:
            continue
        snd_cands = rest[rest["prob_2nd"] >= thr_2nd]
        if len(snd_cands) == 0:
            snd_cands = rest
        idx2 = snd_cands["prob_2nd"].idxmax()
        c2   = rest.loc[idx2, "course"]
        p2   = rest.loc[idx2, "prob_2nd"]

        # 実際の着順
        actual = g.set_index("rank")["course"].to_dict()
        true_1st = actual.get(1, -1)
        true_2nd = actual.get(2, -1)
        hit = (c1 == true_1st and c2 == true_2nd)

        rows.append({
            "date":      date,
            "jyo_cd":    str(jyo_cd).zfill(2),
            "race_no":   race_no,
            "pred_1st":  int(c1),
            "pred_2nd":  int(c2),
            "prob_win":  p1,
            "prob_2nd":  p2,
            "prob_exacta": p1 * p2,   # 2連単確率の近似
            "true_1st":  int(true_1st),
            "true_2nd":  int(true_2nd),
            "hit":       hit,
        })

    return pd.DataFrame(rows)


def attach_payouts(pred_df):
    """払戻データをマージする。"""
    payout_path = os.path.join(RAW_DIR, "race_payouts.csv")
    if not os.path.exists(payout_path):
        print("警告: race_payouts.csv が見つかりません。download_big.py を再実行してください。")
        pred_df["payout_2nd"] = np.nan
        return pred_df

    payouts = pd.read_csv(payout_path, dtype={"jyo_cd": str})
    payouts["jyo_cd"] = payouts["jyo_cd"].str.zfill(2)
    payouts["date"]   = payouts["date"].astype(str)
    pred_df["date"]   = pred_df["date"].astype(str)

    merged = pred_df.merge(
        payouts[["date", "jyo_cd", "race_no", "payout_2nd"]],
        on=["date", "jyo_cd", "race_no"], how="left"
    )
    return merged


def ev_analysis(pred_df):
    """
    EVを計算してEV閾値別のROI・的中率を出力する。

    EV = prob_exacta × expected_payout / 100
       ≈ prob_exacta × (1 / prob_exacta × (1 - take_rate)) = 1 - take_rate   (理論値)

    ただしモデルのprobが精度を持つ場合、高EV = 市場過小評価レースを拾える。
    実際のpayoutを使って計算する。
    """
    # 実際のpayoutがある行のみ
    df = pred_df.dropna(subset=["payout_2nd"]).copy()
    df["ev"] = df["prob_exacta"] * df["payout_2nd"] / 100  # 1単位ベット当たり期待値

    print("\n" + "=" * 70)
    print("期待値 (EV) バックテスト結果")
    print("=" * 70)
    print(f"総レース数: {len(df):,}")
    print(f"全ベット時の的中率: {df['hit'].mean():.1%}")
    print(f"全ベット時のROI:    {(df[df['hit']]['payout_2nd'].sum() / len(df) / 100 - 1):.1%}")

    print(f"\n{'EV閾値':>8} {'ベット数':>8} {'的中数':>8} {'的中率':>8} {'ROI':>10} {'推奨'}")
    print("─" * 60)

    best_roi = -999
    best_threshold = None

    for threshold in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        subset = df[df["ev"] >= threshold]
        if len(subset) == 0:
            continue
        n_bets   = len(subset)
        n_hits   = subset["hit"].sum()
        hit_rate = n_hits / n_bets
        total_return = subset[subset["hit"]]["payout_2nd"].sum()
        cost         = n_bets * 100
        roi          = (total_return - cost) / cost

        marker = ""
        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold
        if roi > 0:
            marker = "← プラス収支"

        print(f"  {threshold:>5.1f}  {n_bets:>8,}  {n_hits:>8,}  {hit_rate:>7.1%}  {roi:>9.1%}  {marker}")

    print(f"\n最適EV閾値: {best_threshold:.1f}  (ROI: {best_roi:.1%})")
    return df, best_threshold


def capital_curve(df, threshold, n_print=20):
    """資金曲線を簡易表示する。"""
    bets = df[df["ev"] >= threshold].sort_values("date").copy()
    if len(bets) == 0:
        return

    capital = 10_000  # 初期資金 10,000円
    unit    = 100     # 1ベット 100円
    history = []
    drawdown_peak = capital
    max_dd = 0

    for _, row in bets.iterrows():
        capital -= unit
        if row["hit"]:
            capital += row["payout_2nd"]
        drawdown_peak = max(drawdown_peak, capital)
        dd = (drawdown_peak - capital) / drawdown_peak
        max_dd = max(max_dd, dd)
        history.append(capital)

    print(f"\n{'=' * 70}")
    print(f"資金曲線 (EV≥{threshold:.1f}, 初期資金10,000円, 1ベット100円)")
    print(f"{'=' * 70}")
    print(f"  総ベット数: {len(bets):,}")
    print(f"  最終資金:   {capital:,.0f}円  ({(capital/10000-1):+.1%})")
    print(f"  最大ドローダウン: {max_dd:.1%}")

    # 月別サマリー
    bets["capital"] = history
    bets["yearmonth"] = pd.to_datetime(bets["date"]).dt.to_period("M")
    monthly = bets.groupby("yearmonth").agg(
        n_bets=("hit", "count"),
        n_hits=("hit", "sum"),
        payout=("payout_2nd", lambda x: x[bets.loc[x.index, "hit"]].sum()),
    )
    monthly["cost"]   = monthly["n_bets"] * unit
    monthly["profit"] = monthly["payout"] - monthly["cost"]
    monthly["roi"]    = monthly["profit"] / monthly["cost"]

    print(f"\n月別損益 (最新{n_print}ヶ月):")
    print(f"  {'月':>8} {'ベット':>6} {'的中':>6} {'損益':>10} {'ROI':>8}")
    print(f"  {'─' * 44}")
    for period, row in monthly.tail(n_print).iterrows():
        sign = "+" if row["profit"] >= 0 else ""
        print(f"  {str(period):>8} {int(row['n_bets']):>6} {int(row['n_hits']):>6} "
              f"{sign}{row['profit']:>8,.0f}円 {row['roi']:>7.1%}")


def main():
    print("=" * 70)
    print("2連単 期待値バックテスト")
    print("=" * 70)

    model_win, model_2nd, feature_cols, thresholds = load_models()

    print("テストデータ読み込み中...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"テストサンプル: {len(test):,}  ({test['date'].min()} ~ {test['date'].max()})")

    print("2連単予測中...")
    pred_df = predict_exacta(test, model_win, model_2nd, feature_cols, thresholds)

    print("払戻データをマージ中...")
    pred_df = attach_payouts(pred_df)

    pred_df, best_threshold = ev_analysis(pred_df)

    capital_curve(pred_df, threshold=best_threshold)

    # 全結果保存
    out_path = os.path.join(CKPT_DIR, "backtest_result.csv")
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n詳細結果CSV: {out_path}")


if __name__ == "__main__":
    main()
