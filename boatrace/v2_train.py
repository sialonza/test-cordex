#!/usr/bin/env python3
"""
v2_train.py
LightGBM + XGBoost アンサンブル学習。

- 3ターゲット: 1着予測, 2着予測, 3着以内予測
- 確率キャリブレーション (Platt scaling)
- レース単位の的中率評価
- 2連単的中率評価

出力: checkpoints/v2/ 以下にモデル・メトリクスを保存
"""

import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import calibration_curve
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v2")
os.makedirs(CKPT_DIR, exist_ok=True)


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "v2_train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "v2_val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "v2_test.csv"))
    with open(os.path.join(DATA_DIR, "v2_feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]
    available = [c for c in feature_cols if c in train.columns]
    return train, val, test, available


def train_lgb(X_tr, y_tr, X_val, y_val, feature_cols):
    """LightGBM 学習"""
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.02,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "max_depth": 8,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(150)],
    )
    return model


def train_xgb(X_tr, y_tr, X_val, y_val, feature_cols):
    """XGBoost 学習"""
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 7,
        "learning_rate": 0.02,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "seed": 42,
        "nthread": -1,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    model = xgb.train(
        params, dtrain, num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=150, verbose_eval=500,
    )
    return model


def calibrate_probabilities(y_val, p_val):
    """
    Platt scaling でキャリブレーション。
    元の確率 p を logit(p) に変換し、ロジスティック回帰で再フィット。
    """
    logit_p = np.log(np.clip(p_val, 1e-6, 1 - 1e-6) / (1 - np.clip(p_val, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logit_p.reshape(-1, 1), y_val)
    return lr


def train_ensemble(train, val, test, feature_cols, target, name):
    """1ターゲットに対するアンサンブル学習"""
    print(f"\n{'━' * 60}")
    print(f"ターゲット: {name} ({target})")
    print(f"{'━' * 60}")

    X_tr = train[feature_cols].values
    y_tr = train[target].values
    X_val = val[feature_cols].values
    y_val = val[target].values
    X_test = test[feature_cols].values
    y_test = test[target].values

    pos_rate = y_tr.mean()
    print(f"Train: {len(X_tr):,} (positive rate: {pos_rate:.1%})")

    # --- LightGBM ---
    print("\n[LightGBM]")
    t0 = time.time()
    lgb_model = train_lgb(X_tr, y_tr, X_val, y_val, feature_cols)
    print(f"  学習時間: {time.time()-t0:.1f}s, best_iter: {lgb_model.best_iteration}")
    p_lgb_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    p_lgb_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    # --- XGBoost ---
    print("\n[XGBoost]")
    t0 = time.time()
    xgb_model = train_xgb(X_tr, y_tr, X_val, y_val, feature_cols)
    print(f"  学習時間: {time.time()-t0:.1f}s, best_iter: {xgb_model.best_iteration}")
    p_xgb_val = xgb_model.predict(
        xgb.DMatrix(X_val, feature_names=feature_cols),
        iteration_range=(0, xgb_model.best_iteration)
    )
    p_xgb_test = xgb_model.predict(
        xgb.DMatrix(X_test, feature_names=feature_cols),
        iteration_range=(0, xgb_model.best_iteration)
    )

    # --- アンサンブル (平均) ---
    p_ens_val = (p_lgb_val + p_xgb_val) / 2
    p_ens_test = (p_lgb_test + p_xgb_test) / 2

    # --- キャリブレーション ---
    print("\n[キャリブレーション]")
    calibrator = calibrate_probabilities(y_val, p_ens_val)
    logit_val = np.log(np.clip(p_ens_val, 1e-6, 1-1e-6) / (1 - np.clip(p_ens_val, 1e-6, 1-1e-6)))
    logit_test = np.log(np.clip(p_ens_test, 1e-6, 1-1e-6) / (1 - np.clip(p_ens_test, 1e-6, 1-1e-6)))
    p_cal_val = calibrator.predict_proba(logit_val.reshape(-1, 1))[:, 1]
    p_cal_test = calibrator.predict_proba(logit_test.reshape(-1, 1))[:, 1]

    # --- 評価 ---
    print(f"\n{'─'*50}")
    print(f"{'メトリクス':>20} {'LGB':>10} {'XGB':>10} {'Ensemble':>10} {'Calibrated':>10}")
    print(f"{'─'*50}")
    for label, y, p_l, p_x, p_e, p_c in [
        ("Val AUC", y_val, p_lgb_val, p_xgb_val, p_ens_val, p_cal_val),
        ("Val LogLoss", y_val, p_lgb_val, p_xgb_val, p_ens_val, p_cal_val),
        ("Test AUC", y_test, p_lgb_test, p_xgb_test, p_ens_test, p_cal_test),
        ("Test LogLoss", y_test, p_lgb_test, p_xgb_test, p_ens_test, p_cal_test),
    ]:
        if "AUC" in label:
            vals = [roc_auc_score(y, p) for p in [p_l, p_x, p_e, p_c]]
        else:
            vals = [log_loss(y, p) for p in [p_l, p_x, p_e, p_c]]
        print(f"{label:>20} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")

    # --- レース単位1着的中率 ---
    if target == "target_win":
        for label, data, probs in [("Val", val, p_cal_val), ("Test", test, p_cal_test)]:
            eval_df = data[["date", "jyo_cd", "race_no", target]].copy()
            eval_df["prob"] = probs
            race_preds = eval_df.groupby(["date", "jyo_cd", "race_no"]).apply(
                lambda g: g.loc[g["prob"].idxmax(), target], include_groups=False
            )
            hit_rate = race_preds.mean()
            print(f"\n  {label} レース単位1着的中率: {hit_rate:.1%} (ランダム: {1/6:.1%})")

    # --- モデル保存 ---
    lgb_model.save_model(os.path.join(CKPT_DIR, f"lgb_{name}.txt"))
    xgb_model.save_model(os.path.join(CKPT_DIR, f"xgb_{name}.json"))
    joblib.dump(calibrator, os.path.join(CKPT_DIR, f"cal_{name}.pkl"))

    # 特徴量重要度
    imp = pd.DataFrame({
        "feature": feature_cols,
        "lgb_gain": lgb_model.feature_importance(importance_type="gain"),
        "xgb_gain": [xgb_model.get_score(importance_type="gain").get(f, 0) for f in feature_cols],
    })
    imp["total"] = imp["lgb_gain"] + imp["xgb_gain"]
    imp = imp.sort_values("total", ascending=False)
    imp.to_csv(os.path.join(CKPT_DIR, f"importance_{name}.csv"), index=False)

    print(f"\n  特徴量重要度 Top 10:")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['feature']:35s} LGB:{r['lgb_gain']:>8.0f}  XGB:{r['xgb_gain']:>8.0f}")

    # メトリクス保存
    metrics = {
        "name": name,
        "target": target,
        "n_features": len(feature_cols),
        "test_auc_lgb": round(roc_auc_score(y_test, p_lgb_test), 4),
        "test_auc_xgb": round(roc_auc_score(y_test, p_xgb_test), 4),
        "test_auc_ens": round(roc_auc_score(y_test, p_ens_test), 4),
        "test_auc_cal": round(roc_auc_score(y_test, p_cal_test), 4),
        "test_logloss_cal": round(log_loss(y_test, p_cal_test), 4),
    }
    with open(os.path.join(CKPT_DIR, f"metrics_{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "lgb": lgb_model, "xgb": xgb_model, "cal": calibrator,
        "metrics": metrics,
    }


def evaluate_exacta(test, models, feature_cols):
    """2連単の的中率を評価"""
    print(f"\n{'═' * 60}")
    print("2連単 (Exacta) 的中率評価")
    print(f"{'═' * 60}")

    X = test[feature_cols].values
    xgb_X = xgb.DMatrix(X, feature_names=feature_cols)

    # 1着確率 (calibrated ensemble)
    win_models = models["win"]
    p_win_lgb = win_models["lgb"].predict(X, num_iteration=win_models["lgb"].best_iteration)
    p_win_xgb = win_models["xgb"].predict(xgb_X, iteration_range=(0, win_models["xgb"].best_iteration))
    p_win_ens = (p_win_lgb + p_win_xgb) / 2
    logit_win = np.log(np.clip(p_win_ens, 1e-6, 1-1e-6) / (1 - np.clip(p_win_ens, 1e-6, 1-1e-6)))
    p_win = win_models["cal"].predict_proba(logit_win.reshape(-1, 1))[:, 1]

    # 2着確率
    snd_models = models["2nd"]
    p_2nd_lgb = snd_models["lgb"].predict(X, num_iteration=snd_models["lgb"].best_iteration)
    p_2nd_xgb = snd_models["xgb"].predict(xgb_X, iteration_range=(0, snd_models["xgb"].best_iteration))
    p_2nd_ens = (p_2nd_lgb + p_2nd_xgb) / 2
    logit_2nd = np.log(np.clip(p_2nd_ens, 1e-6, 1-1e-6) / (1 - np.clip(p_2nd_ens, 1e-6, 1-1e-6)))
    p_2nd = snd_models["cal"].predict_proba(logit_2nd.reshape(-1, 1))[:, 1]

    eval_df = test[["date", "jyo_cd", "race_no", "course", "rank"]].copy()
    eval_df["p_win"] = p_win
    eval_df["p_2nd"] = p_2nd

    results = []
    for (date, jyo, race), g in eval_df.groupby(["date", "jyo_cd", "race_no"]):
        # 1着: p_win最大
        idx1 = g["p_win"].idxmax()
        c1 = g.loc[idx1, "course"]
        p1 = g.loc[idx1, "p_win"]
        # 2着: 1着を除いてp_2nd最大
        rest = g[g.index != idx1]
        if len(rest) == 0:
            continue
        idx2 = rest["p_2nd"].idxmax()
        c2 = rest.loc[idx2, "course"]
        p2 = rest.loc[idx2, "p_2nd"]

        # 条件付き2着確率: P(B=2nd | A=1st) ≈ p_2nd_B / sum(p_2nd for all except A)
        p2_sum = rest["p_2nd"].sum()
        p2_cond = p2 / p2_sum if p2_sum > 0 else p2

        actual = g.set_index("rank")["course"].to_dict()
        hit = (c1 == actual.get(1, -1) and c2 == actual.get(2, -1))
        hit_1st = (c1 == actual.get(1, -1))

        results.append({
            "date": date, "jyo_cd": jyo, "race_no": race,
            "pred_1st": int(c1), "pred_2nd": int(c2),
            "p_win": p1, "p_2nd_cond": p2_cond,
            "p_exacta": p1 * p2_cond,
            "true_1st": actual.get(1, -1), "true_2nd": actual.get(2, -1),
            "hit_1st": hit_1st, "hit": hit,
        })

    rdf = pd.DataFrame(results)
    n = len(rdf)
    print(f"テストレース数: {n:,}")
    print(f"1着的中率: {rdf['hit_1st'].mean():.1%} (ランダム: {1/6:.1%})")
    print(f"2連単的中率: {rdf['hit'].mean():.1%} (ランダム: {1/30:.1%})")
    print(f"  → ランダム比 {rdf['hit'].mean() / (1/30):.1f}倍")

    # 確信度別の的中率
    print(f"\n確率帯別2連単的中率:")
    rdf["prob_bin"] = pd.qcut(rdf["p_exacta"], q=5, labels=False, duplicates="drop")
    for b, g in rdf.groupby("prob_bin"):
        print(f"  確率帯{int(b)+1}: 的中 {g['hit'].mean():.1%} (avg prob: {g['p_exacta'].mean():.1%}, n={len(g)})")

    rdf.to_csv(os.path.join(CKPT_DIR, "exacta_predictions.csv"), index=False)
    return rdf


def main():
    print("=" * 60)
    print("v2 アンサンブル学習 (LightGBM + XGBoost + Calibration)")
    print("=" * 60)

    train, val, test, feature_cols = load_data()
    total = len(train) + len(val) + len(test)
    print(f"総サンプル: {total:,}, 特徴量: {len(feature_cols)}")

    models = {}

    # 1着予測
    models["win"] = train_ensemble(train, val, test, feature_cols, "target_win", "win")

    # 2着予測
    models["2nd"] = train_ensemble(train, val, test, feature_cols, "target_2nd", "2nd")

    # 3着以内予測
    models["top3"] = train_ensemble(train, val, test, feature_cols, "target_top3", "top3")

    # 2連単評価
    exacta_df = evaluate_exacta(test, models, feature_cols)

    # サマリー
    print(f"\n{'═' * 60}")
    print("学習完了サマリー")
    print(f"{'═' * 60}")
    print(f"{'Target':<12} {'LGB AUC':>10} {'XGB AUC':>10} {'Ens AUC':>10} {'Cal AUC':>10}")
    for name in ["win", "2nd", "top3"]:
        m = models[name]["metrics"]
        print(f"{name:<12} {m['test_auc_lgb']:>10.4f} {m['test_auc_xgb']:>10.4f} "
              f"{m['test_auc_ens']:>10.4f} {m['test_auc_cal']:>10.4f}")

    print(f"\nチェックポイント: {CKPT_DIR}")
    print("完了!")


if __name__ == "__main__":
    main()
