#!/usr/bin/env python3
"""
rank_train.py
LightGBM LambdaRank で6艇の相対順位を直接学習する。

通常の binary classification (target_win) との違い:
  - レース内の6艇を1クエリとして扱い、相対的な強さを学習
  - 損失関数が "誰が1位か" だけでなく "全艇の順序" を最適化
  - 出力スコアを softmax でレース内確率に変換して使う

入力: data/processed/train.csv, val.csv, test.csv (per-boat, 6行/レース)
出力: checkpoints/real55k/lgbm_rank.txt
"""

import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "real55k")
os.makedirs(CKPT_DIR, exist_ok=True)

RACE_KEYS = ["date", "jyo_cd", "race_no"]


# ── データ準備 ────────────────────────────────────────────────────────────────

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    available = [c for c in feature_cols if c in train.columns]
    return train, val, test, available


def prepare_ranking_data(df, feature_cols):
    """
    LGBMRanker 用にデータを整形する。

    - レース単位でソート（グループ順序が必要）
    - relevance label = 6 - rank  (rank1→5, rank6→0)
    - group: 各レースの行数 (ほぼ全て6)
    """
    df = df.sort_values(RACE_KEYS + ["course"]).reset_index(drop=True)

    # relevance: 1位が最高スコア
    df["relevance"] = (6 - df["rank"]).clip(lower=0).astype(int)

    X = df[feature_cols].values
    y = df["relevance"].values

    # グループサイズ: 各レースに何艇いるか
    group = df.groupby(RACE_KEYS, sort=False).size().values

    return df, X, y, group


# ── 学習 ──────────────────────────────────────────────────────────────────────

def train_ranker(train, val, feature_cols):
    print("\n" + "=" * 60)
    print("LightGBM LambdaRank 学習")
    print("=" * 60)

    df_train, X_train, y_train, g_train = prepare_ranking_data(train, feature_cols)
    df_val,   X_val,   y_val,   g_val   = prepare_ranking_data(val,   feature_cols)

    print(f"Train: {len(X_train):,} boats / {len(g_train):,} races")
    print(f"Val  : {len(X_val):,} boats / {len(g_val):,} races")
    print(f"特徴量数: {len(feature_cols)}")

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [3, 6],          # NDCG@3 (主指標), NDCG@6
        "boosting_type":    "gbdt",
        "num_leaves":       127,
        "learning_rate":    0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 10,
        "reg_alpha":        0.01,
        "reg_lambda":       0.05,
        "verbose":          -1,
        "seed":             42,
        "n_jobs":           -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train,
                         feature_name=feature_cols)
    dval   = lgb.Dataset(X_val,   label=y_val,   group=g_val,
                         feature_name=feature_cols, reference=dtrain)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=150, first_metric_only=True),
    ]

    print("学習中...")
    start = time.time()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    elapsed = time.time() - start
    print(f"学習時間: {elapsed:.1f}秒 (best_iteration={model.best_iteration})")

    return model, df_val, X_val, y_val, g_val


# ── 評価 ──────────────────────────────────────────────────────────────────────

def softmax_per_race(scores, group):
    """レース内で softmax を適用して確率に変換する。"""
    probs = np.empty_like(scores, dtype=float)
    offset = 0
    for size in group:
        s = scores[offset:offset + size]
        e = np.exp(s - s.max())         # 数値安定
        probs[offset:offset + size] = e / e.sum()
        offset += size
    return probs


def evaluate(model, df, X, y_rel, group, split_name="Test"):
    """
    順位精度を評価する:
      - NDCG@1, NDCG@3 (sklearn)
      - Top-1 的中率 (予測1位が実際の1位)
      - 2連単的中率 (予測1-2位が実際の1-2位)
    """
    scores = model.predict(X, num_iteration=model.best_iteration)
    probs  = softmax_per_race(scores, group)

    df = df.copy()
    df["score"]    = scores
    df["prob_rank"] = probs

    # NDCG (sklearn は 1クエリずつ計算して平均)
    ndcg1_list, ndcg3_list = [], []
    top1_hits, exacta_hits, n_races = 0, 0, 0

    offset = 0
    for size in group:
        sl = slice(offset, offset + size)
        y_true_q = y_rel[sl].reshape(1, -1)
        y_score_q = scores[sl].reshape(1, -1)

        if size >= 2:
            ndcg1_list.append(ndcg_score(y_true_q, y_score_q, k=1))
            k3 = min(3, size)
            ndcg3_list.append(ndcg_score(y_true_q, y_score_q, k=k3))

        # Top-1 的中 (サイズ1のグループはスキップ)
        if size < 2:
            offset += size
            continue

        pred_order = np.argsort(-scores[sl])           # 降順
        true_order = y_rel[sl]

        pred_1st = pred_order[0]
        pred_2nd = pred_order[1]
        true_1st = np.argmax(true_order)               # relevance最大 = rank1
        # rank2 は relevance が2番目 (4)
        true_2nd_cands = np.where(true_order == 4)[0]
        true_2nd = true_2nd_cands[0] if len(true_2nd_cands) > 0 else -1

        if pred_1st == true_1st:
            top1_hits += 1
        if pred_1st == true_1st and pred_2nd == true_2nd:
            exacta_hits += 1
        n_races += 1
        offset += size

    ndcg1 = float(np.mean(ndcg1_list))
    ndcg3 = float(np.mean(ndcg3_list))
    top1_acc   = top1_hits  / n_races
    exacta_acc = exacta_hits / n_races

    print(f"\n--- {split_name} ({n_races:,}レース) ---")
    print(f"NDCG@1   : {ndcg1:.4f}")
    print(f"NDCG@3   : {ndcg3:.4f}")
    print(f"Top-1的中率  : {top1_acc:.4f}  ({top1_acc:.1%})  [ランダム {1/6:.1%}]")
    print(f"2連単的中率  : {exacta_acc:.4f}  ({exacta_acc:.1%})  [ランダム {1/30:.1%}]")

    return {
        "ndcg1": round(ndcg1, 4),
        "ndcg3": round(ndcg3, 4),
        "top1_accuracy": round(top1_acc, 4),
        "exacta_accuracy": round(exacta_acc, 4),
        "n_races": n_races,
    }


# ── バックテスト (EV) ─────────────────────────────────────────────────────────

def backtest_rank(test_df, model, feature_cols, group):
    """
    ランキングモデルで2連単バックテストを実行する。
    払戻データがある場合のみ EV・ROI を計算する。
    """
    payout_path = os.path.join(RAW_DIR, "race_payouts.csv")
    if not os.path.exists(payout_path):
        print("race_payouts.csv が見つかりません。バックテストをスキップします。")
        return

    payouts = pd.read_csv(payout_path, dtype={"jyo_cd": str})
    payouts["jyo_cd"] = payouts["jyo_cd"].str.zfill(2)
    payouts["date"]   = payouts["date"].astype(str)

    df = test_df.copy()
    df["jyo_cd"] = df["jyo_cd"].astype(str).str.zfill(2)
    df["date"]   = df["date"].astype(str)

    X = df[feature_cols].values
    scores = model.predict(X, num_iteration=model.best_iteration)
    probs  = softmax_per_race(scores, group)
    df["score"]     = scores
    df["prob_rank"] = probs

    rows = []
    for (date, jyo, race), g in df.groupby(RACE_KEYS):
        if len(g) < 2:
            continue
        g = g.sort_values("score", ascending=False)
        pred_1st_course = int(g.iloc[0]["course"])
        pred_2nd_course = int(g.iloc[1]["course"])
        prob_win  = float(g.iloc[0]["prob_rank"])
        prob_2nd  = float(g.iloc[1]["prob_rank"])
        prob_exacta = prob_win * prob_2nd

        actual_1st = g.loc[g["rank"] == 1, "course"].values
        actual_2nd = g.loc[g["rank"] == 2, "course"].values
        hit = (
            len(actual_1st) > 0 and len(actual_2nd) > 0
            and pred_1st_course == int(actual_1st[0])
            and pred_2nd_course == int(actual_2nd[0])
        )
        rows.append({
            "date":       date,
            "jyo_cd":     str(jyo).zfill(2),
            "race_no":    race,
            "pred_1st":   pred_1st_course,
            "pred_2nd":   pred_2nd_course,
            "prob_win":   prob_win,
            "prob_2nd":   prob_2nd,
            "prob_exacta": prob_exacta,
            "true_1st":   int(actual_1st[0]) if len(actual_1st) > 0 else -1,
            "true_2nd":   int(actual_2nd[0]) if len(actual_2nd) > 0 else -1,
            "hit":        hit,
        })

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.merge(
        payouts[["date", "jyo_cd", "race_no", "payout_2nd"]],
        on=["date", "jyo_cd", "race_no"], how="left"
    )

    df_pay = pred_df.dropna(subset=["payout_2nd"]).copy()
    df_pay["ev"] = df_pay["prob_exacta"] * df_pay["payout_2nd"] / 100

    print("\n" + "=" * 60)
    print("LambdaRank バックテスト (2連単 EV)")
    print("=" * 60)
    print(f"総レース数: {len(df_pay):,}")
    print(f"全ベット的中率: {df_pay['hit'].mean():.1%}")

    best_roi, best_thr = -999, 0.0
    print(f"\n{'EV閾値':>8} {'ベット':>8} {'的中':>6} {'的中率':>7} {'ROI':>9}")
    print("─" * 45)
    for thr in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
        sub = df_pay[df_pay["ev"] >= thr]
        if len(sub) == 0:
            continue
        cost  = len(sub) * 100
        ret   = sub[sub["hit"]]["payout_2nd"].sum()
        roi   = (ret - cost) / cost
        marker = " ← プラス" if roi > 0 else ""
        print(f"  {thr:>5.1f}  {len(sub):>8,}  {sub['hit'].sum():>6,}  "
              f"{sub['hit'].mean():>6.1%}  {roi:>8.1%}{marker}")
        if roi > best_roi:
            best_roi, best_thr = roi, thr

    print(f"\n最適EV閾値: {best_thr:.1f}  ROI: {best_roi:.1%}")

    # 保存
    out = os.path.join(CKPT_DIR, "rank_backtest_result.csv")
    pred_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"結果保存: {out}")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    train, val, test, feature_cols = load_data()

    # 学習
    model, df_val, X_val, y_val, g_val = train_ranker(train, val, feature_cols)

    # val 評価
    evaluate(model, df_val, X_val, y_val, g_val, split_name="Validation")

    # test 評価
    df_test, X_test, y_test, g_test = prepare_ranking_data(test, feature_cols)
    test_metrics = evaluate(model, df_test, X_test, y_test, g_test, split_name="Test")

    # 特徴量重要度
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\n--- 特徴量重要度 (Top 10) ---")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:10.1f}")

    # モデル保存
    model_path = os.path.join(CKPT_DIR, "lgbm_rank.txt")
    model.save_model(model_path)
    print(f"\nモデル保存: {model_path}")

    importance.to_csv(os.path.join(CKPT_DIR, "lgbm_rank_importance.csv"), index=False)

    metrics = {
        "model_name":    "lgbm_rank",
        "objective":     "lambdarank",
        "best_iteration": model.best_iteration,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "n_features":    len(feature_cols),
    }
    with open(os.path.join(CKPT_DIR, "lgbm_rank_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # EV バックテスト
    backtest_rank(df_test, model, feature_cols, g_test)


if __name__ == "__main__":
    main()
