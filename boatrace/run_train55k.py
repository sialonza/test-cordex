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
    classification_report
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


def train_model(train, val, test, feature_cols, target_col, model_name):
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
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "max_depth": 8,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    # データセット作成
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

    # 学習
    print("学習中...")
    start_time = time.time()

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    elapsed = time.time() - start_time
    print(f"学習時間: {elapsed:.1f}秒 (best_iteration={model.best_iteration})")

    # 予測
    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    # 評価
    print(f"\n--- Validation ---")
    val_auc = roc_auc_score(y_val, y_pred_val)
    val_logloss = log_loss(y_val, y_pred_val)
    val_pred_binary = (y_pred_val > 0.5).astype(int)
    val_acc = accuracy_score(y_val, val_pred_binary)
    print(f"AUC: {val_auc:.4f}")
    print(f"LogLoss: {val_logloss:.4f}")
    print(f"Accuracy: {val_acc:.4f}")

    print(f"\n--- Test ---")
    test_auc = roc_auc_score(y_test, y_pred_test)
    test_logloss = log_loss(y_test, y_pred_test)
    test_pred_binary = (y_pred_test > 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_pred_binary)
    test_precision = precision_score(y_test, test_pred_binary, zero_division=0)
    test_recall = recall_score(y_test, test_pred_binary, zero_division=0)
    test_f1 = f1_score(y_test, test_pred_binary, zero_division=0)

    print(f"AUC: {test_auc:.4f}")
    print(f"LogLoss: {test_logloss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1: {test_f1:.4f}")

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
        "val_auc": round(val_auc, 4),
        "val_logloss": round(val_logloss, 4),
        "val_accuracy": round(val_acc, 4),
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

    return model, metrics


def main():
    print("=" * 60)
    print("ボートレース予測モデル 学習 (LightGBM)")
    print("=" * 60)

    train, val, test, feature_cols = load_data()

    total_samples = len(train) + len(val) + len(test)
    print(f"総サンプル数: {total_samples:,}")
    print(f"特徴量数: {len(feature_cols)}")

    # 1. 1着予測モデル
    model_win, metrics_win = train_model(
        train, val, test, feature_cols,
        target_col="target_win",
        model_name="lgbm_win"
    )

    # 2. 3着以内予測モデル
    model_top3, metrics_top3 = train_model(
        train, val, test, feature_cols,
        target_col="target_top3",
        model_name="lgbm_top3"
    )

    # サマリー
    print("\n" + "=" * 60)
    print("学習完了サマリー")
    print("=" * 60)
    print(f"{'モデル':<20} {'Val AUC':<12} {'Test AUC':<12} {'Test Acc':<12}")
    print(f"{'─' * 56}")
    print(f"{'1着予測':<20} {metrics_win['val_auc']:<12.4f} {metrics_win['test_auc']:<12.4f} {metrics_win['test_accuracy']:<12.4f}")
    print(f"{'3着以内':<20} {metrics_top3['val_auc']:<12.4f} {metrics_top3['test_auc']:<12.4f} {metrics_top3['test_accuracy']:<12.4f}")

    print(f"\nチェックポイント: {CKPT_DIR}")
    print("学習完了!")


if __name__ == "__main__":
    main()
