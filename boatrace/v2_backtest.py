#!/usr/bin/env python3
"""
v2_backtest.py
実際の払戻金データを使ったバックテスト。

戦略:
  1. 均等ベット (全レース・確信度別)
  2. 確信度フィルタリング (p_exacta上位のみベット)
  3. 保守的Kelly (1/8 Kelly, 上限2%)
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v2")


def load_models_and_data():
    lgb_win = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgb_win.txt"))
    xgb_win = xgb.Booster(model_file=os.path.join(CKPT_DIR, "xgb_win.json"))
    cal_win = joblib.load(os.path.join(CKPT_DIR, "cal_win.pkl"))
    lgb_2nd = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgb_2nd.txt"))
    xgb_2nd = xgb.Booster(model_file=os.path.join(CKPT_DIR, "xgb_2nd.json"))
    cal_2nd = joblib.load(os.path.join(CKPT_DIR, "cal_2nd.pkl"))
    with open(os.path.join(DATA_DIR, "v2_feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    test = pd.read_csv(os.path.join(DATA_DIR, "v2_test.csv"), dtype={"jyo_cd": str})
    test["jyo_cd"] = test["jyo_cd"].str.zfill(2)
    payouts = pd.read_csv(os.path.join(RAW_DIR, "race_payouts_real.csv"), dtype={"jyo_cd": str})
    payouts["jyo_cd"] = payouts["jyo_cd"].str.zfill(2)
    return {
        "lgb_win": lgb_win, "xgb_win": xgb_win, "cal_win": cal_win,
        "lgb_2nd": lgb_2nd, "xgb_2nd": xgb_2nd, "cal_2nd": cal_2nd,
        "feature_cols": feature_cols, "test": test, "payouts": payouts,
    }


def predict_calibrated(X, feature_cols, lgb_model, xgb_model, calibrator):
    p_lgb = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    p_xgb = xgb_model.predict(dmat, iteration_range=(0, xgb_model.best_iteration))
    p_ens = (p_lgb + p_xgb) / 2
    logit = np.log(np.clip(p_ens, 1e-6, 1-1e-6) / (1 - np.clip(p_ens, 1e-6, 1-1e-6)))
    return calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]


def build_predictions(ctx):
    test = ctx["test"]
    fc = ctx["feature_cols"]
    available = [c for c in fc if c in test.columns]
    X = test[available].values

    test = test.copy()
    test["p_win"] = predict_calibrated(
        X, available, ctx["lgb_win"], ctx["xgb_win"], ctx["cal_win"])
    test["p_2nd"] = predict_calibrated(
        X, available, ctx["lgb_2nd"], ctx["xgb_2nd"], ctx["cal_2nd"])

    rows = []
    for (date, jyo, race), g in test.groupby(["date", "jyo_cd", "race_no"]):
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

        actual = g.set_index("rank")["course"].to_dict()
        hit = (c1 == actual.get(1, -1) and c2 == actual.get(2, -1))

        rows.append({
            "date": str(date), "jyo_cd": jyo, "race_no": race,
            "pred_1st": int(c1), "pred_2nd": int(c2),
            "p_win": p1, "p_2nd_cond": p2_cond, "p_exacta": p1 * p2_cond,
            "true_1st": actual.get(1, -1), "true_2nd": actual.get(2, -1),
            "hit": hit,
        })

    pred = pd.DataFrame(rows)
    payouts = ctx["payouts"].copy()
    payouts["date"] = payouts["date"].astype(str)
    pred = pred.merge(
        payouts[["date", "jyo_cd", "race_no", "payout_exacta",
                 "payout_trifecta", "payout_win"]],
        on=["date", "jyo_cd", "race_no"], how="left"
    )
    return pred


def flat_bet_analysis(pred):
    """均等ベット (100円) の分析"""
    df = pred.dropna(subset=["payout_exacta"]).copy()
    df["odds"] = df["payout_exacta"] / 100

    print(f"\n{'═' * 70}")
    print("【戦略1】均等ベット分析 (1レース100円)")
    print(f"{'═' * 70}")

    total = len(df)
    hits = df["hit"].sum()
    returns = df[df["hit"]]["payout_exacta"].sum()
    cost = total * 100
    profit = returns - cost
    roi = profit / cost

    print(f"総レース数: {total:,}")
    print(f"的中: {hits:,} ({hits/total:.1%})")
    print(f"投入額: {cost:,}円")
    print(f"回収額: {returns:,.0f}円")
    print(f"損益: {profit:+,.0f}円")
    print(f"ROI: {roi:+.1%}")
    print(f"回収率: {returns/cost:.1%}")

    return df


def confidence_filter_analysis(df):
    """確信度帯別の均等ベット分析"""
    print(f"\n{'═' * 70}")
    print("【戦略2】確信度フィルタリング (p_exacta上位のみベット)")
    print(f"{'═' * 70}")

    # 確信度で5分位
    df = df.copy()
    df["conf_rank"] = df["p_exacta"].rank(pct=True)

    print(f"\n{'フィルタ':>12} {'ベット数':>8} {'的中数':>6} {'的中率':>7} {'平均配当':>8} {'ROI':>8} {'月利':>8}")
    print("─" * 65)

    best_roi = -999
    best_filter = ""

    for label, mask_fn in [
        ("全レース", lambda d: d.index == d.index),  # all
        ("上位80%", lambda d: d["conf_rank"] >= 0.20),
        ("上位60%", lambda d: d["conf_rank"] >= 0.40),
        ("上位40%", lambda d: d["conf_rank"] >= 0.60),
        ("上位20%", lambda d: d["conf_rank"] >= 0.80),
        ("上位10%", lambda d: d["conf_rank"] >= 0.90),
    ]:
        sub = df[mask_fn(df)]
        n = len(sub)
        if n == 0:
            continue
        hits = sub["hit"].sum()
        hit_rate = hits / n
        returns = sub[sub["hit"]]["payout_exacta"].sum()
        cost = n * 100
        roi = (returns - cost) / cost
        avg_payout = sub[sub["hit"]]["payout_exacta"].mean() if hits > 0 else 0

        # テスト期間の日数から月利を推算
        n_days = (pd.to_datetime(df["date"]).max() - pd.to_datetime(df["date"]).min()).days
        monthly_roi = roi * (30 / max(n_days, 1))

        marker = ""
        if roi > best_roi:
            best_roi = roi
            best_filter = label

        print(f"  {label:>10} {n:>8,} {hits:>6,} {hit_rate:>6.1%} {avg_payout:>7,.0f}円 "
              f"{roi:>7.1%} {monthly_roi:>7.1%}")

    print(f"\n  最適フィルタ: {best_filter} (ROI: {best_roi:+.1%})")
    return best_filter


def capital_simulation(df, filter_pct=0.0, initial_capital=100_000):
    """均等ベットの資金シミュレーション"""
    df = df.copy()
    df["conf_rank"] = df["p_exacta"].rank(pct=True)
    bets = df[df["conf_rank"] >= filter_pct].sort_values("date").copy()

    if len(bets) == 0:
        return None

    # ベットサイズ: 資金の1%か500円の小さい方 (保守的)
    capital = initial_capital
    peak = capital
    max_dd = 0
    history = []
    unit = 100  # 1ベット100円

    for _, row in bets.iterrows():
        capital -= unit
        if row["hit"]:
            capital += row["payout_exacta"]
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        history.append(capital)

    bets["capital"] = history

    label = f"上位{int((1-filter_pct)*100)}%" if filter_pct > 0 else "全レース"
    print(f"\n{'═' * 70}")
    print(f"資金推移 ({label}, 1ベット{unit}円, 初期{initial_capital:,}円)")
    print(f"{'═' * 70}")
    print(f"  ベット数: {len(bets):,}")
    print(f"  的中: {bets['hit'].sum():,} ({bets['hit'].mean():.1%})")
    print(f"  最終資金: {capital:,.0f}円 ({(capital/initial_capital - 1):+.1%})")
    print(f"  最大ドローダウン: {max_dd:.1%}")

    # 月別
    bets["yearmonth"] = pd.to_datetime(bets["date"]).dt.to_period("M")
    monthly = []
    for ym, g in bets.groupby("yearmonth"):
        cost = len(g) * unit
        wins = g[g["hit"]]["payout_exacta"].sum()
        monthly.append({
            "month": str(ym), "n_bets": len(g), "n_hits": int(g["hit"].sum()),
            "invested": cost, "returned": wins,
            "profit": wins - cost, "roi": (wins - cost) / cost if cost > 0 else 0,
            "end_capital": g["capital"].iloc[-1],
        })

    mdf = pd.DataFrame(monthly)
    if len(mdf) > 0:
        print(f"\n月別損益:")
        print(f"  {'月':>8} {'ベット':>6} {'的中':>4} {'投入':>10} {'回収':>10} {'損益':>10} {'ROI':>7} {'資金':>12}")
        print(f"  {'─' * 72}")
        for _, r in mdf.iterrows():
            sign = "+" if r["profit"] >= 0 else ""
            print(f"  {r['month']:>8} {int(r['n_bets']):>6} {int(r['n_hits']):>4} "
                  f"{r['invested']:>10,.0f} {r['returned']:>10,.0f} "
                  f"{sign}{r['profit']:>9,.0f} {r['roi']:>6.1%} {r['end_capital']:>11,.0f}")

        win_months = (mdf["profit"] > 0).sum()
        total_profit = mdf["profit"].sum()
        total_invested = mdf["invested"].sum()
        print(f"\n  勝ち月: {win_months}/{len(mdf)} ({win_months/len(mdf):.0%})")
        print(f"  累計損益: {total_profit:+,.0f}円 (ROI: {total_profit/total_invested:+.1%})")
        print(f"  年間換算: {total_profit * 365 / max((pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()).days, 1):+,.0f}円/年")

    return bets


def payout_tier_analysis(df):
    """配当帯別分析: 低配当 vs 高配当"""
    print(f"\n{'═' * 70}")
    print("【戦略3】配当帯別分析")
    print(f"{'═' * 70}")

    df = df.copy()
    bins = [0, 300, 500, 800, 1200, 2000, 5000, float("inf")]
    labels = ["~300", "300~500", "500~800", "800~1.2K", "1.2K~2K", "2K~5K", "5K~"]

    df["payout_bin"] = pd.cut(df["payout_exacta"], bins=bins, labels=labels)

    print(f"\n{'配当帯':>10} {'全体数':>7} {'的中数':>6} {'的中率':>7} {'ROI':>8}")
    print("─" * 45)

    for label in labels:
        sub = df[df["payout_bin"] == label]
        if len(sub) == 0:
            continue
        n = len(sub)
        hits = sub["hit"].sum()
        hit_rate = hits / n
        returns = sub[sub["hit"]]["payout_exacta"].sum()
        cost = n * 100
        roi = (returns - cost) / cost
        print(f"  {label:>8} {n:>7,} {hits:>6,} {hit_rate:>6.1%} {roi:>7.1%}")


def main():
    print("=" * 70)
    print("v2 バックテスト (実払戻金)")
    print("=" * 70)

    print("モデル・データ読み込み中...")
    ctx = load_models_and_data()

    print("2連単予測生成中...")
    pred = build_predictions(ctx)
    payout_n = pred["payout_exacta"].notna().sum()
    print(f"予測: {len(pred):,} レース (払戻データ: {payout_n:,})")

    # 分析
    df = flat_bet_analysis(pred)
    best_filter = confidence_filter_analysis(df)
    payout_tier_analysis(df)

    # 資金シミュレーション
    for filt, label in [(0.0, "全レース"), (0.6, "上位40%"), (0.8, "上位20%")]:
        capital_simulation(df, filter_pct=filt)

    # 結果保存
    pred.to_csv(os.path.join(CKPT_DIR, "backtest_v2.csv"), index=False, encoding="utf-8-sig")
    print(f"\n詳細結果CSV: {os.path.join(CKPT_DIR, 'backtest_v2.csv')}")
    print("バックテスト完了!")


if __name__ == "__main__":
    main()
