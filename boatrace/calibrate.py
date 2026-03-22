#!/usr/bin/env python3
"""
calibrate.py
学習済み LightGBM モデルに確率キャリブレーションを適用する。

手法:
  - Platt scaling    : LogisticRegression(1特徴量) を val 予測値に fitting
  - Isotonic regression : 単調回帰を val 予測値に fitting

評価指標:
  - Log-loss (小さいほど良い)
  - ECE: Expected Calibration Error (小さいほど calibrated)
  - Reliability diagram (10 bin)

出力:
  checkpoints/real55k/lgbm_win_platt.pkl
  checkpoints/real55k/lgbm_win_isotonic.pkl
  checkpoints/real55k/lgbm_2nd_platt.pkl
  checkpoints/real55k/lgbm_2nd_isotonic.pkl
  checkpoints/real55k/calibration_report.json
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "real55k")


# ── ユーティリティ ────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE: キャリブレーション誤差の期待値。
    確率値を n_bins に分け、各ビン内の (平均予測確率 - 実績確率) の加重平均。
    完全に calibrated なら ECE=0。
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc  = y_true[mask].mean()
        ece += mask.mean() * abs(avg_conf - avg_acc)
    return float(ece)


def reliability_report(y_true, y_prob, label="", n_bins=10):
    """Reliability diagram をテキストで出力する。"""
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    print(f"\n  Reliability diagram [{label}]")
    print(f"  {'予測確率':>10}  {'実績確率':>10}  {'差':>8}")
    print(f"  {'─' * 35}")
    for mp, fp in zip(mean_pred, fraction_pos):
        diff = fp - mp
        bar = "▲" if diff > 0.01 else ("▼" if diff < -0.01 else " ")
        print(f"  {mp:>10.3f}  {fp:>10.3f}  {diff:>+7.3f} {bar}")


# ── キャリブレーター ──────────────────────────────────────────────────────────

class PlattCalibrator:
    """
    Platt scaling: val 予測値に LogisticRegression を fitting する。
    sklearn の CalibratedClassifierCV(method='sigmoid') と同等。
    """
    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, y_prob, y_true):
        self.lr.fit(y_prob.reshape(-1, 1), y_true)
        return self

    def predict_proba(self, y_prob):
        return self.lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """
    Isotonic regression: 単調増加の非パラメトリックキャリブレーション。
    """
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_prob, y_true):
        self.iso.fit(y_prob, y_true)
        return self

    def predict_proba(self, y_prob):
        return self.iso.predict(y_prob)


# ── モデル別キャリブレーション ────────────────────────────────────────────────

def calibrate_model(model_name, target_col, model, val_df, test_df, feature_cols):
    """
    1モデルに Platt + Isotonic を適用し、結果を返す。
    """
    available = [c for c in feature_cols if c in val_df.columns]

    X_val  = val_df[available].values
    y_val  = val_df[target_col].values
    X_test = test_df[available].values
    y_test = test_df[target_col].values

    # 元のモデル予測
    p_val  = model.predict(X_val,  num_iteration=model.best_iteration)
    p_test = model.predict(X_test, num_iteration=model.best_iteration)

    # ── ベースライン評価 ──────────────────────────────────────────────────────
    base_logloss = log_loss(y_test, p_test)
    base_ece     = expected_calibration_error(y_test, p_test)
    print(f"\n{'─'*55}")
    print(f"[{model_name}]  ベースライン")
    print(f"  LogLoss : {base_logloss:.5f}")
    print(f"  ECE     : {base_ece:.5f}")
    reliability_report(y_test, p_test, label="Raw")

    results = {
        "model": model_name,
        "base_logloss": round(base_logloss, 5),
        "base_ece":     round(base_ece, 5),
    }

    best_logloss = base_logloss
    best_method  = "none"

    # ── Platt scaling ─────────────────────────────────────────────────────────
    # sklearn の LogisticRegression を直接保存 (pickle互換性のため)
    platt_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt_lr.fit(p_val.reshape(-1, 1), y_val)
    p_platt_test = platt_lr.predict_proba(p_test.reshape(-1, 1))[:, 1]
    platt_logloss = log_loss(y_test, p_platt_test)
    platt_ece     = expected_calibration_error(y_test, p_platt_test)

    print(f"\n  Platt scaling (val fitting)")
    print(f"  LogLoss : {platt_logloss:.5f}  ({platt_logloss - base_logloss:+.5f})")
    print(f"  ECE     : {platt_ece:.5f}  ({platt_ece - base_ece:+.5f})")
    reliability_report(y_test, p_platt_test, label="Platt")

    platt_path = os.path.join(CKPT_DIR, f"{model_name}_platt.pkl")
    joblib.dump(platt_lr, platt_path)
    print(f"  保存: {platt_path}")

    results["platt_logloss"] = round(platt_logloss, 5)
    results["platt_ece"]     = round(platt_ece, 5)

    if platt_logloss < best_logloss:
        best_logloss = platt_logloss
        best_method  = "platt"

    # ── Isotonic regression ───────────────────────────────────────────────────
    # sklearn の IsotonicRegression を直接保存
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(p_val, y_val)
    p_iso_test = np.clip(iso_reg.predict(p_test), 1e-7, 1 - 1e-7)
    iso_logloss = log_loss(y_test, p_iso_test)
    iso_ece     = expected_calibration_error(y_test, p_iso_test)

    print(f"\n  Isotonic regression (val fitting)")
    print(f"  LogLoss : {iso_logloss:.5f}  ({iso_logloss - base_logloss:+.5f})")
    print(f"  ECE     : {iso_ece:.5f}  ({iso_ece - base_ece:+.5f})")
    reliability_report(y_test, p_iso_test, label="Isotonic")

    iso_path = os.path.join(CKPT_DIR, f"{model_name}_isotonic.pkl")
    joblib.dump(iso_reg, iso_path)
    print(f"  保存: {iso_path}")

    results["iso_logloss"] = round(iso_logloss, 5)
    results["iso_ece"]     = round(iso_ece, 5)

    if iso_logloss < best_logloss:
        best_logloss = iso_logloss
        best_method  = "isotonic"

    results["best_method"]   = best_method
    results["best_logloss"]  = round(best_logloss, 5)

    print(f"\n  ★ 最良手法: {best_method}  (LogLoss {best_logloss:.5f})")

    return results


# ── 2連単的中率: キャリブレーション後の評価 ──────────────────────────────────

def _apply_cal(cal, raw_prob):
    """LogisticRegression / IsotonicRegression どちらでも使えるヘルパー。"""
    if cal is None:
        return raw_prob
    if hasattr(cal, "predict_proba"):                    # LogisticRegression
        return cal.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
    return np.clip(cal.predict(raw_prob), 1e-7, 1 - 1e-7)  # IsotonicRegression

def eval_exacta_with_calibrator(test_df, model_win, model_2nd,
                                 cal_win, cal_2nd, feature_cols, label=""):
    """
    calibrator を通した prob_win / prob_2nd で 2連単的中率を計算する。
    """
    available = [c for c in feature_cols if c in test_df.columns]
    X = test_df[available].values

    p_win_raw = model_win.predict(X, num_iteration=model_win.best_iteration)
    p_2nd_raw = model_2nd.predict(X, num_iteration=model_2nd.best_iteration)

    p_win = _apply_cal(cal_win, p_win_raw)
    p_2nd = _apply_cal(cal_2nd, p_2nd_raw)

    df = test_df[["date", "jyo_cd", "race_no", "course", "rank"]].copy()
    df["prob_win"] = p_win
    df["prob_2nd"] = p_2nd

    hits = []
    for _, g in df.groupby(["date", "jyo_cd", "race_no"]):
        pred_1st = g.loc[g["prob_win"].idxmax(), "course"]
        rest     = g[g["course"] != pred_1st]
        if len(rest) == 0:
            continue
        pred_2nd = rest.loc[rest["prob_2nd"].idxmax(), "course"]
        a1 = g.loc[g["rank"] == 1, "course"].values
        a2 = g.loc[g["rank"] == 2, "course"].values
        hit = (len(a1) > 0 and len(a2) > 0
               and pred_1st == a1[0] and pred_2nd == a2[0])
        hits.append(hit)

    rate = sum(hits) / len(hits) if hits else 0
    print(f"  2連単的中率 [{label}]: {rate:.4f} ({rate:.1%})")
    return rate


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("確率キャリブレーション (Platt / Isotonic)")
    print("=" * 60)

    # データ読み込み
    val  = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    # モデル読み込み
    model_win = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_win.txt"))
    model_2nd = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_2nd.txt"))

    print(f"Val  : {len(val):,} samples")
    print(f"Test : {len(test):,} samples")
    print(f"特徴量: {len(feature_cols)}")

    # キャリブレーション実行
    report = {}
    report["lgbm_win"] = calibrate_model(
        "lgbm_win", "target_win", model_win, val, test, feature_cols
    )
    report["lgbm_2nd"] = calibrate_model(
        "lgbm_2nd", "target_2nd", model_2nd, val, test, feature_cols
    )

    # レポート保存
    report_path = os.path.join(CKPT_DIR, "calibration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n\nレポート保存: {report_path}")

    # ── キャリブレーション後の 2連単的中率比較 ──────────────────────────────
    print("\n" + "=" * 60)
    print("2連単的中率 比較 (キャリブレーション前後)")
    print("=" * 60)

    # Raw (キャリブレーションなし)
    eval_exacta_with_calibrator(
        test, model_win, model_2nd, None, None, feature_cols, label="Raw"
    )

    # Platt
    cal_win_platt = joblib.load(os.path.join(CKPT_DIR, "lgbm_win_platt.pkl"))
    cal_2nd_platt = joblib.load(os.path.join(CKPT_DIR, "lgbm_2nd_platt.pkl"))
    eval_exacta_with_calibrator(
        test, model_win, model_2nd, cal_win_platt, cal_2nd_platt, feature_cols,
        label="Platt"
    )

    # Isotonic
    cal_win_iso = joblib.load(os.path.join(CKPT_DIR, "lgbm_win_isotonic.pkl"))
    cal_2nd_iso = joblib.load(os.path.join(CKPT_DIR, "lgbm_2nd_isotonic.pkl"))
    eval_exacta_with_calibrator(
        test, model_win, model_2nd, cal_win_iso, cal_2nd_iso, feature_cols,
        label="Isotonic"
    )

    # ── サマリー ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("サマリー")
    print("=" * 60)
    for mname, r in report.items():
        print(f"\n{mname}:")
        print(f"  Base     LogLoss={r['base_logloss']:.5f}  ECE={r['base_ece']:.5f}")
        print(f"  Platt    LogLoss={r['platt_logloss']:.5f}  ECE={r['platt_ece']:.5f}"
              f"  ({r['platt_logloss']-r['base_logloss']:+.5f})")
        print(f"  Isotonic LogLoss={r['iso_logloss']:.5f}  ECE={r['iso_ece']:.5f}"
              f"  ({r['iso_logloss']-r['base_logloss']:+.5f})")
        print(f"  → 最良: {r['best_method']}")

    print("\nキャリブレーション完了!")
    print(f"保存先: {CKPT_DIR}/lgbm_{{win,2nd}}_{{platt,isotonic}}.pkl")


if __name__ == "__main__":
    main()
