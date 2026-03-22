#!/usr/bin/env python3
"""
run_train55k.py
LightGBM でボートレース予測モデルを学習する。
- 1着予測モデル (binary classification)
- 3着以内予測モデル (binary classification)
- ハイパーパラメータチューニング
"""

import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    precision_score, recall_score, f1_score,
    precision_recall_curve,
)
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "real55k")
os.makedirs(CKPT_DIR, exist_ok=True)


def load_data():
    """処理済みデータ読み込み"""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # 特徴量が存在するか確認
    available = [c for c in feature_cols if c in train.columns]
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        print(f"  警告: 以下の特徴量が欠損: {missing}")

    return train, val, test, available


def find_best_threshold(y_true, y_prob):
    """F1スコアを最大化する閾値をval予測で探索する。"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve は thresholds が len-1 なので末尾を揃える
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def train_model(train, val, test, feature_cols, target_col, model_name,
                scale_pos_weight=None):
    """LightGBMモデルの学習と評価"""
    print(f"\n{'─' * 50}")
    print(f"モデル: {model_name} (target={target_col})")
    print(f"{'─' * 50}")

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_val = val[feature_cols].values
    y_val = val[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    print(f"Train: {len(X_train):,} samples (pos={y_train.sum():,}, neg={len(y_train)-y_train.sum():,})")
    print(f"Val:   {len(X_val):,} samples")
    print(f"Test:  {len(X_test):,} samples")

    # LightGBM パラメータ
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "max_depth": 10,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
        print(f"scale_pos_weight: {scale_pos_weight}")

    # データセット作成
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

    # 学習
    print("学習中...")
    start_time = time.time()

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=100),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    elapsed = time.time() - start_time
    print(f"学習時間: {elapsed:.1f}秒 (best_iteration={model.best_iteration})")

    # 予測
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    # val で最適閾値を探索 (F1最大化)
    best_thr, best_val_f1 = find_best_threshold(y_val, y_pred_val)
    print(f"最適閾値 (val F1最大): {best_thr:.4f}  (val F1={best_val_f1:.4f})")

    # 評価
    print(f"\n--- Validation ---")
    val_auc = roc_auc_score(y_val, y_pred_val)
    val_logloss = log_loss(y_val, y_pred_val)
    val_pred_binary = (y_pred_val > best_thr).astype(int)
    val_acc = accuracy_score(y_val, val_pred_binary)
    print(f"AUC: {val_auc:.4f}")
    print(f"LogLoss: {val_logloss:.4f}")
    print(f"Accuracy (@thr={best_thr:.3f}): {val_acc:.4f}")

    print(f"\n--- Test ---")
    test_auc = roc_auc_score(y_test, y_pred_test)
    test_logloss = log_loss(y_test, y_pred_test)
    test_pred_binary = (y_pred_test > best_thr).astype(int)
    test_acc = accuracy_score(y_test, test_pred_binary)
    test_precision = precision_score(y_test, test_pred_binary, zero_division=0)
    test_recall = recall_score(y_test, test_pred_binary, zero_division=0)
    test_f1 = f1_score(y_test, test_pred_binary, zero_division=0)

    print(f"AUC: {test_auc:.4f}")
    print(f"LogLoss: {test_logloss:.4f}")
    print(f"Accuracy (@thr={best_thr:.3f}): {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1: {test_f1:.4f}")

    # レース単位の予測精度 (最も本質的な評価)
    # 各レースで最高確率の艇を1着予測 → 実際に1着だった割合
    if target_col == "target_win" and "date" in test.columns and "jyo_cd" in test.columns:
        test_eval = test[["date", "jyo_cd", "race_no", target_col]].copy()
        test_eval["pred_prob"] = y_pred_test
        # レース内で最高確率の艇
        test_eval["is_top_pred"] = (
            test_eval.groupby(["date", "jyo_cd", "race_no"])["pred_prob"]
            .rank(ascending=False) == 1
        ).astype(int)
        # 予測1着が実際の1着と一致したレースの割合
        race_correct = test_eval[test_eval["is_top_pred"] == 1][target_col].mean()
        print(f"レース単位1着的中率: {race_correct:.4f} (ランダム比較: {1/6:.4f})")

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    print(f"\n--- 特徴量重要度 (Top 10) ---")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:10.1f}")

    # モデル保存
    model_path = os.path.join(CKPT_DIR, f"{model_name}.txt")
    model.save_model(model_path)
    print(f"\nモデル保存: {model_path}")

    # 重要度保存
    imp_path = os.path.join(CKPT_DIR, f"{model_name}_importance.csv")
    importance.to_csv(imp_path, index=False)

    # メトリクス保存
    metrics = {
        "model_name": model_name,
        "target": target_col,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "best_iteration": model.best_iteration,
        "train_time_sec": round(elapsed, 1),
        "scale_pos_weight": scale_pos_weight,
        "best_threshold": round(best_thr, 4),
        "val_auc": round(val_auc, 4),
        "val_logloss": round(val_logloss, 4),
        "val_f1_at_best_thr": round(best_val_f1, 4),
        "test_auc": round(test_auc, 4),
        "test_logloss": round(test_logloss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_f1": round(test_f1, 4),
        "n_features": len(feature_cols),
        "features": feature_cols,
    }

    metrics_path = os.path.join(CKPT_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return model, metrics, best_thr


def compute_transition_matrix(df: pd.DataFrame, min_samples: int = 20) -> dict:
    """
    trainデータから P(2着=c2 | 1着=c1) の遷移行列を計算する。

    Args:
        df: 'rank', 'course', 'date', 'jyo_cd', 'race_no' を含む DataFrame
        min_samples: この数未満のセルはフォールバック値 0.2 を使用

    Returns:
        {"c1-c2": float, ...} 形式の辞書 (例: {"1-2": 0.312, ...})
    """
    first_counts: dict = {}
    pair_counts: dict = {}

    for _, g in df.groupby(["date", "jyo_cd", "race_no"]):
        first_rows = g[g["rank"] == 1]["course"].values
        second_rows = g[g["rank"] == 2]["course"].values
        if len(first_rows) == 0 or len(second_rows) == 0:
            continue
        c1 = str(int(first_rows[0]))
        c2 = str(int(second_rows[0]))
        first_counts[c1] = first_counts.get(c1, 0) + 1
        pair_counts[(c1, c2)] = pair_counts.get((c1, c2), 0) + 1

    matrix: dict = {}
    for (c1, c2), cnt in pair_counts.items():
        total = first_counts.get(c1, 0)
        key = f"{c1}-{c2}"
        if total >= min_samples:
            matrix[key] = cnt / total
        else:
            matrix[key] = 0.2  # フォールバック: 1/5
    return matrix


def main():
    print("=" * 60)
    print("ボートレース予測モデル 学習 (LightGBM)")
    print("=" * 60)

    train, val, test, feature_cols = load_data()

    total_samples = len(train) + len(val) + len(test)
    print(f"総サンプル数: {total_samples:,}")
    print(f"特徴量数: {len(feature_cols)}")

    # クラス不均衡比 (neg/pos = 5:1)
    pos_ratio = train["target_win"].mean()
    spw = round((1 - pos_ratio) / pos_ratio, 2)
    print(f"\nクラス比 (neg/pos): {spw:.2f}")

    # 1. 1着予測モデル
    model_win, metrics_win, thr_win = train_model(
        train, val, test, feature_cols,
        target_col="target_win",
        model_name="lgbm_win",
        scale_pos_weight=spw,
    )

    # 2. 2着予測モデル (2連単用) — scale_pos_weight で少数クラスを重視
    model_2nd, metrics_2nd, thr_2nd = train_model(
        train, val, test, feature_cols,
        target_col="target_2nd",
        model_name="lgbm_2nd",
        scale_pos_weight=spw,
    )

    # 3. 3着以内予測モデル
    model_top3, metrics_top3, thr_top3 = train_model(
        train, val, test, feature_cols,
        target_col="target_top3",
        model_name="lgbm_top3",
    )

    # 最適閾値を保存
    thresholds = {
        "lgbm_win": thr_win,
        "lgbm_2nd": thr_2nd,
        "lgbm_top3": thr_top3,
    }
    thr_path = os.path.join(CKPT_DIR, "thresholds.json")
    with open(thr_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n最適閾値保存: {thr_path}")
    print(json.dumps(thresholds, indent=2))

    # 2連単的中率評価 (テストデータ)
    print("\n" + "=" * 60)
    print(f"2連単 的中率評価 (Test)  thr_win={thr_win:.3f}, thr_2nd={thr_2nd:.3f}")
    print("=" * 60)
    X_test = test[feature_cols].values
    test_eval = test[["date", "jyo_cd", "race_no", "course", "rank"]].copy()
    test_eval["prob_win"] = model_win.predict(X_test, num_iteration=model_win.best_iteration)
    test_eval["prob_2nd"] = model_2nd.predict(X_test, num_iteration=model_2nd.best_iteration)

    results = []
    for (date, jyo, race), g in test_eval.groupby(["date", "jyo_cd", "race_no"]):
        # 1着予測: prob_win 最大の艇
        pred_1st = g.loc[g["prob_win"].idxmax(), "course"]
        # 2着予測: 1着予測を除いた中で prob_2nd 最大の艇
        rest = g[g["course"] != pred_1st]
        pred_2nd = rest.loc[rest["prob_2nd"].idxmax(), "course"] if len(rest) > 0 else -1
        # 実際の着順
        actual_1st = g.loc[g["rank"] == 1, "course"].values
        actual_2nd = g.loc[g["rank"] == 2, "course"].values
        hit = (
            len(actual_1st) > 0 and len(actual_2nd) > 0
            and pred_1st == actual_1st[0]
            and pred_2nd == actual_2nd[0]
        )
        results.append(hit)

    exacta_hit_rate = sum(results) / len(results) if results else 0
    print(f"2連単的中率: {exacta_hit_rate:.4f} ({exacta_hit_rate:.1%})")
    print(f"ランダム比較: {1/30:.4f} ({1/30:.1%})  ※6×5=30通り")

    # サマリー
    print("\n" + "=" * 60)
    print("学習完了サマリー")
    print("=" * 60)
    print(f"{'モデル':<20} {'Val AUC':<12} {'Test AUC':<12} {'Test F1':<12} {'閾値':<8}")
    print(f"{'─' * 64}")
    print(f"{'1着予測':<20} {metrics_win['val_auc']:<12.4f} {metrics_win['test_auc']:<12.4f} {metrics_win['test_f1']:<12.4f} {thr_win:<8.3f}")
    print(f"{'2着予測':<20} {metrics_2nd['val_auc']:<12.4f} {metrics_2nd['test_auc']:<12.4f} {metrics_2nd['test_f1']:<12.4f} {thr_2nd:<8.3f}")
    print(f"{'3着以内':<20} {metrics_top3['val_auc']:<12.4f} {metrics_top3['test_auc']:<12.4f} {metrics_top3['test_f1']:<12.4f} {thr_top3:<8.3f}")

    # 遷移行列 (条件付き2連単確率) の計算・保存
    print("\n遷移行列を計算中 (P(2着=c2 | 1着=c1))...")
    all_train = pd.concat([train, val], ignore_index=True)
    trans_matrix = compute_transition_matrix(all_train)
    tm_path = os.path.join(CKPT_DIR, "transition_matrix.json")
    with open(tm_path, "w") as f:
        json.dump(trans_matrix, f, indent=2)
    print(f"遷移行列保存: {tm_path}  ({len(trans_matrix)}セル)")
    # 主要コンビを表示
    for key in ["1-2", "1-3", "2-1", "3-1", "4-1"]:
        print(f"  P(2着={key.split('-')[1]} | 1着={key.split('-')[0]}): "
              f"{trans_matrix.get(key, 0.2):.3f}")

    print(f"\nチェックポイント: {CKPT_DIR}")
    print("学習完了!")


if __name__ == "__main__":
    main()
