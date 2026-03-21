#!/usr/bin/env python3
"""
v3_train.py
実戦で勝つためのモデル訓練。

戦略:
  1. 各コースの1着確率をモデルで予測 → P_model(c)
  2. 過去の平均払戻から市場の暗黙的確率を推定 → P_market(c) = 100/avg_payout
  3. Edge = P_model - P_market > threshold のときに賭ける
  4. ケリー基準でベット額を決定

モデル: LightGBM (コース別バイナリ分類 + 確率キャリブレーション)
"""
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM未インストール。sklearn fallbackを使用します。")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

class PlattCalibratedModel:
    """LightGBM + Platt scaling (picklable)"""
    def __init__(self, base, platt):
        self.base = base
        self.platt = platt
    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.platt.predict_proba(raw)


DATA_DIR  = Path(__file__).parent / "data"
PROC_DIR  = DATA_DIR / "processed"
CKPT_DIR  = Path(__file__).parent / "checkpoints" / "v3"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(prefix='v3'):
    train = pd.read_csv(PROC_DIR / f'{prefix}_train.csv', parse_dates=['date'])
    val   = pd.read_csv(PROC_DIR / f'{prefix}_val.csv',   parse_dates=['date'])
    test  = pd.read_csv(PROC_DIR / f'{prefix}_test.csv',  parse_dates=['date'])
    with open(PROC_DIR / f'{prefix}_feature_cols.txt') as f:
        feat_cols = [l.strip() for l in f if l.strip()]
    return train, val, test, feat_cols


def make_xy(df, feat_cols, course):
    """コースcが1着 = 1, それ以外 = 0"""
    X = df[feat_cols].values.astype(np.float32)
    y = (df['winner_course'] == course).astype(int).values
    return X, y


def train_course_model(train, val, feat_cols, course):
    """1コースの勝ち負け分類器を訓練"""
    X_tr, y_tr = make_xy(train, feat_cols, course)
    X_va, y_va = make_xy(val,   feat_cols, course)

    # 欠損を中央値で補完
    med = np.nanmedian(X_tr, axis=0)
    for j in range(X_tr.shape[1]):
        X_tr[np.isnan(X_tr[:, j]), j] = med[j]
        X_va[np.isnan(X_va[:, j]), j] = med[j]

    pos_rate = y_tr.mean()
    scale_pos = (1 - pos_rate) / max(pos_rate, 1e-6)

    if HAS_LGB:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'scale_pos_weight': scale_pos,
            'verbose': -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)]
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        )
        model.fit(X_tr, y_tr)

    # 確率キャリブレーション (Platt scaling: valセットでLogisticRegressionを当てる)
    raw_va = model.predict_proba(X_va)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(raw_va, y_va)

    cal = PlattCalibratedModel(model, platt)
    y_pred = cal.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, y_pred)
    ll  = log_loss(y_va, y_pred)
    print(f"  コース{course}: AUC={auc:.4f}, LogLoss={ll:.4f}")

    return cal, med


def train_all(prefix='v3'):
    print("データロード中...")
    train, val, test, feat_cols = load_split(prefix)

    print(f"Train:{len(train):,}, Val:{len(val):,}, Test:{len(test):,}")
    print(f"特徴量:{len(feat_cols)}")

    models, medians = {}, {}
    print("\n=== コース別モデル訓練 ===")
    for course in range(1, 7):
        model, med = train_course_model(train, val, feat_cols, course)
        models[course] = model
        medians[course] = med

    # 保存
    joblib.dump({'models': models, 'medians': medians, 'feat_cols': feat_cols},
                CKPT_DIR / 'course_models.pkl')
    print(f"\nモデル保存: {CKPT_DIR}/course_models.pkl")

    # バリデーション評価
    print("\n=== バリデーション評価 ===")
    eval_ev(val, feat_cols, models, medians, split_name='Validation')

    print("\n=== テスト評価 ===")
    eval_ev(test, feat_cols, models, medians, split_name='Test')

    return models, medians, feat_cols


def predict_proba_all(df, feat_cols, models, medians):
    """各レースの各コースの勝ち確率を返す"""
    proba = np.zeros((len(df), 6))
    for i, course in enumerate(range(1, 7)):
        X, _ = make_xy(df, feat_cols, course)
        med = medians[course]
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = med[j]
        proba[:, i] = models[course].predict_proba(X)[:, 1]

    # 行ごとに合計1に正規化
    row_sum = proba.sum(axis=1, keepdims=True).clip(min=1e-8)
    proba = proba / row_sum
    return proba  # shape: (n_races, 6)


def compute_ev(proba, df):
    """
    期待値 = P_model × payout - 1  (賭け金1単位あたり)
    正ならプラス期待値 (買い時)
    """
    ev_win = np.full(len(df), np.nan)
    ev_exacta = np.full(len(df), np.nan)

    if 'payout_win' in df.columns:
        # 単勝: 1着予測コースの確率 × 実際の払戻
        pred_course = np.argmax(proba, axis=1) + 1  # 1-6
        for i, row in enumerate(df.itertuples()):
            c = pred_course[i]
            p = proba[i, c - 1]
            payout = getattr(row, 'payout_win', np.nan)
            if not np.isnan(payout) and payout > 0:
                ev_win[i] = p * (payout / 100) - 1

    if 'payout_exacta' in df.columns:
        # 2連単: 上位2コースの確率積 × 実際の払戻
        sorted_idx = np.argsort(-proba, axis=1)
        for i, row in enumerate(df.itertuples()):
            c1 = sorted_idx[i, 0] + 1
            c2 = sorted_idx[i, 1] + 1
            p12 = proba[i, c1 - 1] * proba[i, c2 - 1] / max(1 - proba[i, c1 - 1], 1e-6)
            payout = getattr(row, 'payout_exacta', np.nan)
            if not np.isnan(payout) and payout > 0:
                ev_exacta[i] = p12 * (payout / 100) - 1

    return ev_win, ev_exacta


def eval_ev(df, feat_cols, models, medians, split_name=''):
    """期待値ベースのバックテスト評価"""
    proba = predict_proba_all(df, feat_cols, models, medians)
    ev_win, ev_exacta = compute_ev(proba, df)

    # 単勝精度
    pred_winner = np.argmax(proba, axis=1) + 1
    actual_winner = df['winner_course'].values
    top1_acc = (pred_winner == actual_winner).mean()

    # コース1の確率
    c1_prob_mean = proba[:, 0].mean()

    print(f"{split_name}:")
    print(f"  1着予測精度: {top1_acc:.3f} (ランダム=0.167, コース1基準=?)")
    print(f"  コース1平均予測確率: {c1_prob_mean:.3f}")

    # EV分析 (実際の払戻があるデータのみ)
    ev_valid = ev_win[~np.isnan(ev_win)]
    if len(ev_valid) > 0:
        print(f"  単勝EV: 平均={ev_valid.mean():.4f}, 正率={( ev_valid > 0).mean():.3f}")
        print(f"  単勝EV>0のレース: {(ev_valid > 0).sum():,}/{len(ev_valid):,}")

    ev_valid2 = ev_exacta[~np.isnan(ev_exacta)]
    if len(ev_valid2) > 0:
        print(f"  2連単EV: 平均={ev_valid2.mean():.4f}, 正率={(ev_valid2 > 0).mean():.3f}")

    return proba, ev_win, ev_exacta


def load_model(prefix='v3'):
    data = joblib.load(CKPT_DIR / 'course_models.pkl')
    return data['models'], data['medians'], data['feat_cols']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='v3')
    args = parser.parse_args()
    train_all(args.prefix)
