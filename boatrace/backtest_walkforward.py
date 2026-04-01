#!/usr/bin/env python3
"""
backtest_walkforward.py
テストデータを月次フォールドに分割し、時系列順に評価する Walk-forward バックテスト。

【リーク防止の保証】
  - 特徴量はすべて convert_boatracecsv.py で生成済み（rolling は closed='left'）
  - テストデータは学習データより必ず時間的に後
  - 本スクリプトはテストデータのみを使用し、追加の特徴量計算は行わない

【評価指標 (各月次フォールド)】
  - AUC         : 1着予測の識別性能
  - LogLoss     : 確率の精度
  - PR-AUC      : 不均衡データに強い AUC
  - F1 (最適閾値): 適合率・再現率のバランス
  - 2連単的中率  : 実務的命中精度
  - ROI (EV≥1.0): 期待値フィルタ後の収益率

出力:
  checkpoints/real55k/walkforward_folds.csv   # フォールド別指標
  checkpoints/real55k/walkforward_summary.json # 統計サマリー
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import (
    roc_auc_score, log_loss,
    average_precision_score, f1_score, precision_recall_curve,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "real55k")

TAKE_RATE        = 0.225   # 控除率
WINNER_THRESHOLD = 0.25    # スタッキング用フラグ閾値
MIN_FOLD_RACES   = 200     # フォールドの最小レース数（小さすぎるフォールドを除外）


# ── ユーティリティ ──────────────────────────────────────────────────────────

def load_assets():
    """モデル・特徴量・キャリブレーター・遷移行列を読み込む。"""
    model_win = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_win.txt"))
    model_2nd = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_2nd.txt"))

    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    stacking_path = os.path.join(CKPT_DIR, "stacking_feature_cols.txt")
    if os.path.exists(stacking_path):
        with open(stacking_path) as f:
            stacking_cols = [l.strip() for l in f if l.strip()]
    else:
        stacking_cols = None

    # キャリブレーター (isotonic 優先)
    calibrators = {}
    for mname in ("lgbm_win", "lgbm_2nd"):
        for method in ("isotonic", "platt"):
            p = os.path.join(CKPT_DIR, f"{mname}_{method}.pkl")
            if os.path.exists(p) and mname not in calibrators:
                calibrators[mname] = joblib.load(p)
                print(f"  キャリブレーター読み込み: {mname} ({method})")
                break

    tm_path = os.path.join(CKPT_DIR, "transition_matrix.json")
    trans_matrix = {}
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            trans_matrix = json.load(f)

    with open(os.path.join(CKPT_DIR, "thresholds.json")) as f:
        thresholds = json.load(f)

    return model_win, model_2nd, feature_cols, stacking_cols, calibrators, trans_matrix, thresholds


def apply_calibrator(cal, raw_prob):
    """キャリブレーターを適用する（なければ raw をそのまま返す）。"""
    if cal is None:
        return raw_prob
    if hasattr(cal, "predict_proba"):
        return cal.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
    return np.clip(cal.predict(raw_prob), 1e-7, 1 - 1e-7)


def best_f1_threshold(y_true, y_prob):
    """PR曲線から F1 を最大化する閾値を返す。"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) == 0, 0,
            2 * precision * recall / (precision + recall)
        )
    idx = np.argmax(f1[:-1])  # 最後の要素は precision=1, recall=0 で除外
    return float(thresholds[idx]), float(f1[idx])


# ── 予測 ────────────────────────────────────────────────────────────────────

def run_inference(df, model_win, model_2nd, feature_cols, stacking_cols, calibrators):
    """
    df に prob_win / prob_2nd 列を付与して返す。
    スタッキング特徴量の計算もここで行う。
    """
    df = df.copy()
    avail_win = [c for c in feature_cols if c in df.columns]

    raw_win = model_win.predict(df[avail_win].values,
                                num_iteration=model_win.best_iteration)
    df["prob_win_raw"] = raw_win

    # スタッキング特徴量
    grp = df.groupby(["date", "jyo_cd", "race_no"])["prob_win_raw"]
    df["prob_win_max_in_race"] = grp.transform("max")
    df["prob_win_rank"]        = grp.rank(ascending=False)
    df["is_likely_winner"]     = (df["prob_win_raw"] >= WINNER_THRESHOLD).astype(int)

    if stacking_cols is not None:
        avail_2nd = [c for c in stacking_cols if c in df.columns]
    else:
        avail_2nd = avail_win + ["prob_win_raw", "prob_win_max_in_race",
                                  "prob_win_rank", "is_likely_winner"]
        avail_2nd = [c for c in avail_2nd if c in df.columns]

    raw_2nd = model_2nd.predict(df[avail_2nd].values,
                                num_iteration=model_2nd.best_iteration)

    df["prob_win"] = apply_calibrator(calibrators.get("lgbm_win"), raw_win)
    df["prob_2nd"] = apply_calibrator(calibrators.get("lgbm_2nd"), raw_2nd)

    return df


# ── フォールド評価 ───────────────────────────────────────────────────────────

def evaluate_fold(fold_df, trans_matrix, thr_win):
    """
    1フォールドの評価指標を計算する。
    返り値: dict (metrics) + pred_df (予測結果)
    """
    # ── モデル指標 (選手単位) ─────────────────────────────────────────────
    y_true_win = fold_df["target_win"].values
    y_prob_win = fold_df["prob_win"].values

    auc     = roc_auc_score(y_true_win, y_prob_win)
    logloss = log_loss(y_true_win, y_prob_win)
    pr_auc  = average_precision_score(y_true_win, y_prob_win)
    opt_thr, f1 = best_f1_threshold(y_true_win, y_prob_win)

    # ── 2連単予測 (レース単位) ────────────────────────────────────────────
    rows = []
    for (date, jyo_cd, race_no), g in fold_df.groupby(["date", "jyo_cd", "race_no"]):
        if len(g) < 2:
            continue

        # 1着予測
        win_cands = g[g["prob_win"] >= thr_win]
        if len(win_cands) == 0:
            win_cands = g
        idx1  = win_cands["prob_win"].idxmax()
        c1    = int(g.loc[idx1, "course"])
        p1    = float(g.loc[idx1, "prob_win"])

        # 2着予測
        rest = g[g.index != idx1]
        idx2 = rest["prob_2nd"].idxmax()
        c2   = int(rest.loc[idx2, "course"])
        p2   = float(rest.loc[idx2, "prob_2nd"])

        # 遷移確率
        trans_key  = f"{c1}-{c2}"
        trans_prob = trans_matrix.get(trans_key, 0.2)
        prob_exacta = p1 * trans_prob

        # 実績
        actual = g.set_index("rank")["course"].to_dict()
        true_1st = int(actual.get(1, -1))
        true_2nd = int(actual.get(2, -1))
        hit = (c1 == true_1st and c2 == true_2nd)

        rows.append({
            "date":         str(date),
            "jyo_cd":       str(jyo_cd).zfill(2),
            "race_no":      int(race_no),
            "pred_1st":     c1,
            "pred_2nd":     c2,
            "prob_win":     p1,
            "prob_2nd":     p2,
            "prob_exacta":  round(prob_exacta, 6),
            "true_1st":     true_1st,
            "true_2nd":     true_2nd,
            "hit":          hit,
        })

    if not rows:
        return None, pd.DataFrame()

    pred_df = pd.DataFrame(rows)

    # ── 払戻マージ ────────────────────────────────────────────────────────
    payout_path = os.path.join(RAW_DIR, "race_payouts.csv")
    if os.path.exists(payout_path):
        payouts = pd.read_csv(payout_path, dtype={"jyo_cd": str})
        payouts["jyo_cd"] = payouts["jyo_cd"].str.zfill(2)
        payouts["date"]   = payouts["date"].astype(str)
        pred_df = pred_df.merge(
            payouts[["date", "jyo_cd", "race_no", "payout_2nd"]],
            on=["date", "jyo_cd", "race_no"], how="left"
        )
    else:
        pred_df["payout_2nd"] = np.nan

    # ── ROI 計算 (EV≥1.0 フィルタ) ───────────────────────────────────────
    df_with_payout = pred_df.dropna(subset=["payout_2nd"]).copy()
    df_with_payout["ev"] = df_with_payout["prob_exacta"] * df_with_payout["payout_2nd"] / 100

    for ev_thr in (0.0, 1.0):
        subset = df_with_payout[df_with_payout["ev"] >= ev_thr]
        if len(subset) > 0:
            n_bets       = len(subset)
            n_hits       = subset["hit"].sum()
            total_return = subset[subset["hit"]]["payout_2nd"].sum()
            cost         = n_bets * 100
            roi          = (total_return - cost) / cost
        else:
            n_bets = n_hits = 0
            roi    = float("nan")

        suffix = "_all" if ev_thr == 0.0 else "_ev1"
        pred_df[f"n_bets{suffix}"]  = n_bets
        pred_df[f"n_hits{suffix}"]  = n_hits
        pred_df[f"roi{suffix}"]     = roi

    # ── 指標まとめ ────────────────────────────────────────────────────────
    n_races     = len(pred_df)
    hit_rate    = pred_df["hit"].mean()
    n_bets_all  = pred_df["n_bets_all"].iloc[0] if n_races else 0
    n_hits_all  = pred_df["n_hits_all"].iloc[0] if n_races else 0
    roi_all     = pred_df["roi_all"].iloc[0] if n_races else float("nan")
    n_bets_ev1  = pred_df["n_bets_ev1"].iloc[0] if n_races else 0
    n_hits_ev1  = pred_df["n_hits_ev1"].iloc[0] if n_races else 0
    roi_ev1     = pred_df["roi_ev1"].iloc[0] if n_races else float("nan")

    metrics = {
        "n_races":      n_races,
        "auc":          round(auc, 5),
        "logloss":      round(logloss, 5),
        "pr_auc":       round(pr_auc, 5),
        "f1":           round(f1, 5),
        "f1_threshold": round(opt_thr, 5),
        "hit_rate":     round(hit_rate, 5),
        "n_bets_all":   int(n_bets_all),
        "n_hits_all":   int(n_hits_all),
        "roi_all":      round(roi_all, 5) if not np.isnan(roi_all) else None,
        "n_bets_ev1":   int(n_bets_ev1),
        "n_hits_ev1":   int(n_hits_ev1),
        "roi_ev1":      round(roi_ev1, 5) if not np.isnan(roi_ev1) else None,
    }
    return metrics, pred_df


# ── Walk-forward ループ ──────────────────────────────────────────────────────

def walkforward(test_df, model_win, model_2nd, feature_cols, stacking_cols,
                calibrators, trans_matrix, thresholds, fold_unit="M"):
    """
    テストデータを fold_unit 単位（デフォルト月次）に分割して評価する。

    Parameters
    ----------
    fold_unit : str
        pandas Period frequency string. "M" = 月次, "W" = 週次.
    """
    thr_win = thresholds.get("lgbm_win", 0.5)

    test_df = test_df.copy()
    test_df["date"] = pd.to_datetime(test_df["date"])
    test_df["period"] = test_df["date"].dt.to_period(fold_unit)
    test_df["date"] = test_df["date"].dt.strftime("%Y%m%d")

    periods = sorted(test_df["period"].unique())
    print(f"\nフォールド数: {len(periods)}  ({periods[0]} ~ {periods[-1]})")

    fold_results = []
    all_preds    = []

    for i, period in enumerate(periods):
        fold_raw = test_df[test_df["period"] == period].copy()
        n_races  = fold_raw.groupby(["date", "jyo_cd", "race_no"]).ngroups

        if n_races < MIN_FOLD_RACES:
            print(f"  [{i+1:03d}] {period}  レース数が少ないためスキップ ({n_races} races)")
            continue

        # 推論
        fold_df = run_inference(fold_raw, model_win, model_2nd,
                                feature_cols, stacking_cols, calibrators)

        metrics, pred_df = evaluate_fold(fold_df, trans_matrix, thr_win)

        if metrics is None:
            continue

        metrics["period"] = str(period)
        fold_results.append(metrics)
        all_preds.append(pred_df)

        roi_ev1_str = (f"{metrics['roi_ev1']:+.1%}" if metrics["roi_ev1"] is not None
                       else "  n/a ")
        print(
            f"  [{i+1:03d}] {period}  "
            f"races={n_races:4d}  "
            f"AUC={metrics['auc']:.4f}  "
            f"LogLoss={metrics['logloss']:.4f}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"hit={metrics['hit_rate']:.3%}  "
            f"ROI(EV≥1)={roi_ev1_str}"
        )

    return fold_results, pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


# ── サマリー ─────────────────────────────────────────────────────────────────

def summarize(fold_results):
    """フォールド一覧から統計サマリーを計算する。"""
    df = pd.DataFrame(fold_results)

    def stats(col):
        s = df[col].dropna()
        return {
            "mean":   round(float(s.mean()), 5),
            "std":    round(float(s.std()), 5),
            "min":    round(float(s.min()), 5),
            "max":    round(float(s.max()), 5),
            "q25":    round(float(s.quantile(0.25)), 5),
            "median": round(float(s.median()), 5),
            "q75":    round(float(s.quantile(0.75)), 5),
        }

    summary = {
        "n_folds":   len(fold_results),
        "n_folds_roi_ev1_positive": int((df["roi_ev1"].dropna() > 0).sum()),
        "metrics":   {
            "auc":      stats("auc"),
            "logloss":  stats("logloss"),
            "pr_auc":   stats("pr_auc"),
            "f1":       stats("f1"),
            "hit_rate": stats("hit_rate"),
            "roi_all":  stats("roi_all"),
            "roi_ev1":  stats("roi_ev1"),
        },
    }
    return summary


def print_summary(summary):
    print("\n" + "=" * 75)
    print("Walk-forward バックテスト サマリー")
    print("=" * 75)
    print(f"フォールド数:              {summary['n_folds']}")
    print(f"ROI(EV≥1) プラス月数:      {summary['n_folds_roi_ev1_positive']}")
    print()
    print(f"{'指標':>12}  {'平均':>8}  {'std':>8}  {'中央値':>8}  {'最小':>8}  {'最大':>8}")
    print("─" * 65)
    for name, s in summary["metrics"].items():
        print(
            f"  {name:>10}  {s['mean']:>8.5f}  {s['std']:>8.5f}  "
            f"{s['median']:>8.5f}  {s['min']:>8.5f}  {s['max']:>8.5f}"
        )


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("Walk-forward 時系列バックテスト (月次フォールド)")
    print("=" * 75)

    print("アセット読み込み中...")
    (model_win, model_2nd, feature_cols, stacking_cols,
     calibrators, trans_matrix, thresholds) = load_assets()

    print("テストデータ読み込み中...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"テストサンプル: {len(test):,}  ({test['date'].min()} ~ {test['date'].max()})")

    fold_results, all_preds = walkforward(
        test, model_win, model_2nd, feature_cols, stacking_cols,
        calibrators, trans_matrix, thresholds, fold_unit="M"
    )

    if not fold_results:
        print("評価可能なフォールドがありません。")
        return

    summary = summarize(fold_results)
    print_summary(summary)

    # 保存
    folds_path   = os.path.join(CKPT_DIR, "walkforward_folds.csv")
    summary_path = os.path.join(CKPT_DIR, "walkforward_summary.json")

    pd.DataFrame(fold_results).to_csv(folds_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nフォールド別指標: {folds_path}")
    print(f"サマリーJSON:     {summary_path}")
    print("\nWalk-forward バックテスト完了!")


if __name__ == "__main__":
    main()
