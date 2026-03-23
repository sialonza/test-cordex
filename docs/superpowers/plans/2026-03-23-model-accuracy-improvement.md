# モデル精度改善 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 特徴量拡充・スタッキング・Optunaチューニングにより2連単ROIを+28.4%から+35%以上に改善する

**Architecture:** convert_boatracecsv.py に直近フォーム・対戦相手特徴量を追加し、run_train55k.py でOptunaチューニング後にスタッキング（lgbm_win出力をlgbm_2ndの入力に使う）を実装。predict_today.py も同じスタッキング順序に更新。

**Tech Stack:** Python, LightGBM, Optuna 4.8.0, pandas, numpy, scikit-learn

---

## File Map

| ファイル | 変更種別 | 変更内容 |
|---|---|---|
| `boatrace/convert_boatracecsv.py` | Modify | recent_form・venue_form マージ、対戦相手特徴量追加 |
| `boatrace/run_train55k.py` | Modify | Optunaチューニング追加、スタッキング実装 |
| `boatrace/predict_today.py` | Modify | recent_form・venue_form マージ、対戦相手特徴量、スタッキング順序更新 |
| `boatrace/tests/test_features.py` | Create | 特徴量生成・スタッキングのユニットテスト（両方含む） |

---

## Task 1: テスト環境セットアップ

**Files:**
- Create: `boatrace/tests/__init__.py`
- Create: `boatrace/tests/test_features.py`

- [ ] **Step 1: testsディレクトリを作成**

```bash
mkdir -p /home/user/test-cordex/boatrace/tests
touch /home/user/test-cordex/boatrace/tests/__init__.py
```

- [ ] **Step 2: 特徴量テストの雛形を作成**

`boatrace/tests/test_features.py` を作成:

```python
"""特徴量エンジニアリングのユニットテスト"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_race_df(n_races=3, n_courses=6):
    """テスト用の最小レースDataFrameを生成"""
    records = []
    for race_no in range(1, n_races + 1):
        for course in range(1, n_courses + 1):
            records.append({
                "date": "2026-01-01",
                "jyo_cd": "01",
                "race_no": race_no,
                "course": course,
                "racer_id": 1000 + (race_no - 1) * 6 + course,
                "win_rate": float(course),
                "nirenritsu": float(course * 10),
                "sanrenritsu": float(course * 15),
                "motor_nirenritsu": 35.0,
                "st": 0.15,
                "tenji_time": 6.75,
                "weather": "晴",
                "wind_speed": 3,
                "wave": 4,
                "water_temp": 14.0,
                "racer_class": "B1",
                "rank": course,  # course番目が course着
            })
    return pd.DataFrame(records)


class TestOpponentFeatures:
    def test_opponent_features_shape(self):
        """対戦相手特徴量が正しく生成されること"""
        from convert_boatracecsv import add_opponent_features
        df = make_race_df(n_races=2)
        result = add_opponent_features(df)
        expected_cols = [
            "opponent_avg_win_rate", "opponent_max_win_rate",
            "win_rate_vs_best", "opponent_avg_class", "relative_form_rank"
        ]
        for col in expected_cols:
            assert col in result.columns, f"列 '{col}' が見つかりません"

    def test_opponent_avg_win_rate_correct(self):
        """opponent_avg_win_rateがコース1に対して正しく計算されること"""
        from convert_boatracecsv import add_opponent_features
        df = make_race_df(n_races=1)
        result = add_opponent_features(df)
        # コース1の選手: win_rate=1.0, 対戦相手は win_rate=2,3,4,5,6 → 平均=4.0
        course1_row = result[result["course"] == 1].iloc[0]
        assert abs(course1_row["opponent_avg_win_rate"] - 4.0) < 1e-6

    def test_no_nan_in_opponent_features(self):
        """対戦相手特徴量にNaNがないこと"""
        from convert_boatracecsv import add_opponent_features
        df = make_race_df(n_races=3)
        result = add_opponent_features(df)
        cols = ["opponent_avg_win_rate", "opponent_max_win_rate", "win_rate_vs_best"]
        for col in cols:
            assert result[col].isna().sum() == 0, f"'{col}' にNaNがあります"


class TestStackingFeatures:
    def test_stacking_features_shape(self):
        """スタッキング特徴量が正しく生成されること"""
        from run_train55k import add_stacking_features
        df = make_race_df(n_races=2)
        df["prob_win"] = np.random.rand(len(df))
        result = add_stacking_features(df)
        expected_cols = ["prob_win_max_in_race", "prob_win_rank", "is_likely_winner"]
        for col in expected_cols:
            assert col in result.columns, f"列 '{col}' が見つかりません"

    def test_prob_win_rank_is_1_to_6(self):
        """prob_win_rankが1〜6の値を持つこと"""
        from run_train55k import add_stacking_features
        df = make_race_df(n_races=2)
        df["prob_win"] = np.linspace(0.05, 0.30, len(df))
        result = add_stacking_features(df)
        ranks = result["prob_win_rank"].unique()
        assert set(ranks).issubset(set(range(1, 7)))

    def test_is_likely_winner_uses_threshold(self):
        """is_likely_winner が WINNER_THRESHOLD 以上の艇にフラグを立てること"""
        from run_train55k import add_stacking_features, WINNER_THRESHOLD
        df = make_race_df(n_races=1)
        # コース1は0.10、コース6は0.40（閾値0.25より高い）
        probs = [0.10, 0.15, 0.20, 0.20, 0.15, 0.40]
        df["prob_win"] = probs * 1  # n_races=1
        result = add_stacking_features(df)
        # prob_win >= 0.25 の艇だけフラグが1
        expected = (df["prob_win"] >= WINNER_THRESHOLD).astype(int).tolist()
        assert result["is_likely_winner"].tolist() == expected
```

- [ ] **Step 3: テストが失敗することを確認（実装前）**

```bash
cd /home/user/test-cordex/boatrace && python -m pytest tests/test_features.py -v 2>&1 | head -30
```

Expected: `ImportError` または `AttributeError`（まだ実装していないため）

- [ ] **Step 4: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/tests/ && git commit -m "test: 特徴量・スタッキングのテスト雛形を追加"
```

---

## Task 2: convert_boatracecsv.py — recent_form・venue_form マージ追加

**Files:**
- Modify: `boatrace/convert_boatracecsv.py`

- [ ] **Step 1: load_and_merge() に recent_form マージを追加**

`convert_boatracecsv.py` の `load_and_merge()` 内、`jyo_course_stats` マージの直後（約128行目）に追加:

```python
    # 直近フォームマージ
    recent_form_path = os.path.join(EXTRA_DIR, "racer_recent_form.csv")
    if os.path.exists(recent_form_path):
        recent_form = pd.read_csv(recent_form_path)
        df = df.merge(recent_form, on="racer_id", how="left")
        print(f"  直近フォームをマージ")

    # 会場別フォームマージ
    venue_form_path = os.path.join(EXTRA_DIR, "racer_venue_form.csv")
    if os.path.exists(venue_form_path):
        venue_form = pd.read_csv(venue_form_path)
        df["jyo_cd"] = df["jyo_cd"].astype(str).str.zfill(2)
        venue_form["jyo_cd"] = venue_form["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(venue_form, on=["racer_id", "jyo_cd"], how="left")
        print(f"  会場別フォームをマージ")
```

- [ ] **Step 2: add_opponent_features() 関数を追加**

`engineer_features()` 関数の直前（約141行目）に新関数を追加:

```python
def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース内の対戦相手特徴量を計算する。
    各艇について「同レースの他5艇の情報」から相対的な強さを表す特徴量を生成する。
    """
    df = df.copy()
    race_group = ["date", "jyo_cd", "race_no"]

    # 対戦相手の平均勝率 = (レース合計 - 自分) / 5
    race_sum_win = df.groupby(race_group)["win_rate"].transform("sum")
    race_count   = df.groupby(race_group)["win_rate"].transform("count")
    df["opponent_avg_win_rate"] = (race_sum_win - df["win_rate"]) / (race_count - 1).clip(lower=1)

    # 対戦相手の最強選手の勝率
    race_max_win = df.groupby(race_group)["win_rate"].transform("max")
    df["opponent_max_win_rate"] = np.where(
        df["win_rate"] == race_max_win,
        df.groupby(race_group)["win_rate"].transform(lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else x.max()),
        race_max_win,
    )

    # 自分の勝率 ÷ 最強対戦相手の勝率 (0除算防止)
    df["win_rate_vs_best"] = df["win_rate"] / df["opponent_max_win_rate"].clip(lower=0.1)

    # 対戦相手の平均クラス
    class_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    if "racer_class_num" not in df.columns:
        df["racer_class_num"] = df["racer_class"].map(class_map).fillna(0) if "racer_class" in df.columns else 0
    race_sum_class = df.groupby(race_group)["racer_class_num"].transform("sum")
    df["opponent_avg_class"] = (race_sum_class - df["racer_class_num"]) / (race_count - 1).clip(lower=1)

    # 直近フォームのレース内順位 (form_win_5 があれば使用、なければ win_rate)
    form_col = "form_win_5" if "form_win_5" in df.columns else "win_rate"
    df["relative_form_rank"] = df.groupby(race_group)[form_col].rank(ascending=False)

    return df
```

- [ ] **Step 3: engineer_features() 内で add_opponent_features を呼ぶ**

`engineer_features()` 内、`# ターゲット` のコードブロック直前（約200行目）に追加:

```python
    # 対戦相手特徴量
    df = add_opponent_features(df)
```

- [ ] **Step 4: feature_cols リストに新特徴量を追加**

`main()` 内の `feature_cols` リスト（約253行目）を更新。`"mkt_win_prob_180d",` の後に追加:

```python
        # 直近フォーム
        "form_win_3", "form_top2_3", "form_avg_rank_3", "form_avg_st_3",
        "form_win_5", "form_top2_5", "form_avg_rank_5", "form_avg_st_5",
        "form_win_10", "form_top2_10", "form_avg_rank_10", "form_avg_st_10",
        "form_trend_win", "form_trend_rank",
        "form_win_streak", "form_no_win_streak",
        # 会場別フォーム
        "form_venue_win_5", "form_venue_top2_5",
        # 対戦相手特徴量
        "opponent_avg_win_rate", "opponent_max_win_rate",
        "win_rate_vs_best", "opponent_avg_class", "relative_form_rank",
```

- [ ] **Step 5: テストを実行して対戦相手特徴量が通ることを確認**

```bash
cd /home/user/test-cordex/boatrace && python -m pytest tests/test_features.py::TestOpponentFeatures -v
```

Expected: 3テストすべて PASS

- [ ] **Step 6: convert_boatracecsv.py を実行して特徴量数が増えたことを確認**

```bash
cd /home/user/test-cordex/boatrace && python convert_boatracecsv.py 2>&1 | grep -E "特徴量数|使用特徴量"
```

Expected: `特徴量数: 8X` (44より増加)、`使用特徴量 (6X個):`

- [ ] **Step 7: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/convert_boatracecsv.py && git commit -m "feat: 直近フォーム・会場別フォーム・対戦相手特徴量を追加"
```

---

## Task 3: run_train55k.py — Optunaチューニング追加

**Files:**
- Modify: `boatrace/run_train55k.py`

- [ ] **Step 1: import に optuna を追加**

ファイル先頭の import ブロック（約14行目）に追加:

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

- [ ] **Step 2: tune_hyperparams() 関数を追加**

`train_model()` 関数の直前に追加:

```python
def tune_hyperparams(X_train, y_train, X_val, y_val,
                     feature_cols, scale_pos_weight=None,
                     n_trials=50) -> dict:
    """
    Optuna で LightGBM のハイパーパラメータを最適化する。
    最適化指標: val AUC 最大化。
    """
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
        }
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_cols, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=callbacks,
        )
        return model.best_score["val"]["auc"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update({
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    })
    if scale_pos_weight is not None:
        best["scale_pos_weight"] = scale_pos_weight
    print(f"  Optuna best val AUC: {study.best_value:.4f}  (試行={n_trials})")
    return best
```

- [ ] **Step 3: train_model() の params 固定値をチューニング結果に差し替え**

`train_model()` 内の `# LightGBM パラメータ` ブロック（約73〜93行目）を以下で置き換え:

```python
    # Optunaでハイパーパラメータをチューニング
    print(f"  Optunaチューニング中 (50試行)...")
    best_params = tune_hyperparams(
        X_train, y_train, X_val, y_val,
        feature_cols=feature_cols,
        scale_pos_weight=scale_pos_weight,
        n_trials=50,
    )
    params = best_params
```

また、`train_model()` の metrics dict（約188行目）に以下を追加して、Optunaのベストパラメータを保存できるようにする:

```python
        "optuna_best_params": {
            k: v for k, v in best_params.items()
            if k not in ("objective", "metric", "boosting_type",
                         "bagging_freq", "verbose", "seed", "n_jobs",
                         "scale_pos_weight")
        },
```

- [ ] **Step 4: optuna_params.json を保存する処理を main() に追加**

`main()` 内の `# 最適閾値を保存` の直前（約293行目）に追加:

```python
    # Optunaベストパラメータを保存
    # tune_hyperparams() が返す best dict を metrics 経由で保存する
    # （model.params は LightGBM 内部デフォルト値も含むため使わない）
    optuna_params = {
        "lgbm_win":  metrics_win.get("optuna_best_params", {}),
        "lgbm_2nd":  metrics_2nd.get("optuna_best_params", {}),
        "lgbm_top3": metrics_top3.get("optuna_best_params", {}),
    }
    with open(os.path.join(CKPT_DIR, "optuna_params.json"), "w") as f:
        json.dump(optuna_params, f, indent=2)
    print(f"Optunaパラメータ保存: {CKPT_DIR}/optuna_params.json")
```

- [ ] **Step 5: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/run_train55k.py && git commit -m "feat: Optunaハイパーパラメータチューニングを追加"
```

---

## Task 4: run_train55k.py — スタッキング実装

**Files:**
- Modify: `boatrace/run_train55k.py`

- [ ] **Step 1: add_stacking_features() 関数を追加**

`compute_transition_matrix()` 関数の直前に追加:

```python
WINNER_THRESHOLD = 0.25  # この確率以上を「1着候補」とみなす


def add_stacking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    lgbm_win の予測確率を使ったスタッキング特徴量を生成する。
    df には 'prob_win', 'date', 'jyo_cd', 'race_no' 列が必要。
    """
    df = df.copy()
    race_group = ["date", "jyo_cd", "race_no"]

    # レース内での1着確率の最大値
    df["prob_win_max_in_race"] = df.groupby(race_group)["prob_win"].transform("max")

    # 1着確率のレース内順位 (1=最高確率)
    df["prob_win_rank"] = df.groupby(race_group)["prob_win"].rank(ascending=False)

    # 1着候補フラグ: 1着確率が WINNER_THRESHOLD 以上の艇
    df["is_likely_winner"] = (df["prob_win"] >= WINNER_THRESHOLD).astype(int)

    return df
```

- [ ] **Step 2: スタッキング特徴量テストが通ることを確認**

```bash
cd /home/user/test-cordex/boatrace && python -m pytest tests/test_features.py::TestStackingFeatures -v
```

Expected: 3テストすべて PASS

- [ ] **Step 3: main() のスタッキングフローを実装**

`main()` 内の `# 2. 2着予測モデル` ブロック（約277〜283行目）を以下で置き換え:

```python
    # --- スタッキング: lgbm_win の確率を lgbm_2nd の入力に追加 ---
    stacking_feature_cols = feature_cols + ["prob_win", "prob_win_max_in_race",
                                             "prob_win_rank", "is_likely_winner"]

    def attach_win_probs(split_df, model):
        """分割データに prob_win とスタッキング特徴量を付与"""
        X = split_df[feature_cols].values
        split_df = split_df.copy()
        split_df["prob_win"] = model.predict(X, num_iteration=model.best_iteration)
        split_df = add_stacking_features(split_df)
        return split_df

    train_stacked = attach_win_probs(train, model_win)
    val_stacked   = attach_win_probs(val,   model_win)
    test_stacked  = attach_win_probs(test,  model_win)

    # スタッキング特徴量が存在するものだけ使用
    available_stacking = [c for c in stacking_feature_cols if c in train_stacked.columns]

    # 2. 2着予測モデル (スタッキング特徴量使用)
    model_2nd, metrics_2nd, thr_2nd = train_model(
        train_stacked, val_stacked, test_stacked, available_stacking,
        target_col="target_2nd",
        model_name="lgbm_2nd",
        scale_pos_weight=spw,
    )

    # スタッキング特徴量リストを保存 (predict_today.py で参照)
    stacking_cols_path = os.path.join(CKPT_DIR, "stacking_feature_cols.txt")
    with open(stacking_cols_path, "w") as f:
        f.write("\n".join(available_stacking))
```

- [ ] **Step 4: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/run_train55k.py boatrace/tests/ && git commit -m "feat: lgbm_winの出力を使ったスタッキング予測を実装"
```

---

## Task 5: predict_today.py — 新特徴量・スタッキング対応

**Files:**
- Modify: `boatrace/predict_today.py`

- [ ] **Step 1: prepare_features() に recent_form・venue_form・対戦相手特徴量を追加**

`prepare_features()` 内、`# ─── レース内相対特徴量` ブロックの直前（約127行目）に追加:

```python
    # 直近フォームマージ
    recent_form_path = os.path.join(EXTRA_DIR, "racer_recent_form.csv")
    if os.path.exists(recent_form_path):
        recent_form = pd.read_csv(recent_form_path)
        df = df.merge(recent_form, on="racer_id", how="left")

    # 会場別フォームマージ
    venue_form_path = os.path.join(EXTRA_DIR, "racer_venue_form.csv")
    if os.path.exists(venue_form_path):
        venue_form = pd.read_csv(venue_form_path)
        df["jyo_cd"] = df["jyo_cd"].astype(str).str.zfill(2)
        venue_form["jyo_cd"] = venue_form["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(venue_form, on=["racer_id", "jyo_cd"], how="left")

    # 対戦相手特徴量 (df["date"] は prepare_features() 冒頭で変換済み)
    race_group = ["date", "jyo_cd", "race_no"]
    race_sum_win  = df.groupby(race_group)["win_rate"].transform("sum")
    race_count    = df.groupby(race_group)["win_rate"].transform("count")
    df["opponent_avg_win_rate"] = (race_sum_win - df["win_rate"]) / (race_count - 1).clip(lower=1)
    race_max_win  = df.groupby(race_group)["win_rate"].transform("max")
    df["opponent_max_win_rate"] = np.where(
        df["win_rate"] == race_max_win,
        df.groupby(race_group)["win_rate"].transform(
            lambda x: x.nlargest(2).iloc[-1] if len(x) > 1 else x.max()
        ),
        race_max_win,
    )
    df["win_rate_vs_best"] = df["win_rate"] / df["opponent_max_win_rate"].clip(lower=0.1)
    df["opponent_avg_class"] = (
        df.groupby(race_group)["racer_class_num"].transform("sum") - df["racer_class_num"]
    ) / (race_count - 1).clip(lower=1)
    form_col = "form_win_5" if "form_win_5" in df.columns else "win_rate"
    df["relative_form_rank"] = df.groupby(race_group)[form_col].rank(ascending=False)
```

- [ ] **Step 2: main() の予測フローをスタッキング順序に更新**

まず `predict_today.py` の import ブロック末尾に以下を追加して `WINNER_THRESHOLD` を共有:

```python
from run_train55k import WINNER_THRESHOLD
```

次に `main()` 内の `# 予測` ブロック（約200〜205行目）を以下で置き換え:

```python
    # 特徴量リスト（通常 + スタッキング）
    stacking_cols_path = os.path.join(CKPT_DIR, "stacking_feature_cols.txt")
    if os.path.exists(stacking_cols_path):
        with open(stacking_cols_path) as f:
            stacking_feature_cols = [line.strip() for line in f if line.strip()]
    else:
        stacking_feature_cols = feature_cols  # フォールバック

    # Step 1: lgbm_win で1着確率を予測
    available_win = [c for c in feature_cols if c in df.columns]
    X_win = df[available_win].values
    df["prob_win"] = model_win.predict(X_win, num_iteration=model_win.best_iteration)

    # Step 2: スタッキング特徴量を計算 (WINNER_THRESHOLD は run_train55k からimport済み)
    race_group = ["date", "jyo_cd", "race_no"]
    df["prob_win_max_in_race"] = df.groupby(race_group)["prob_win"].transform("max")
    df["prob_win_rank"] = df.groupby(race_group)["prob_win"].rank(ascending=False)
    df["is_likely_winner"] = (df["prob_win"] >= WINNER_THRESHOLD).astype(int)

    # Step 3: lgbm_2nd で2着確率を予測（スタッキング特徴量込み）
    available_2nd = [c for c in stacking_feature_cols if c in df.columns]
    X_2nd = df[available_2nd].values
    df["prob_2nd"] = model_2nd.predict(X_2nd, num_iteration=model_2nd.best_iteration)
```

- [ ] **Step 3: predict_today.py を実行して動作確認**

```bash
cd /home/user/test-cordex/boatrace && python predict_today.py 2>&1 | head -40
```

Expected: エラーなく予測結果が出力される

- [ ] **Step 4: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/predict_today.py && git commit -m "feat: predict_today.pyに新特徴量・スタッキング対応を追加"
```

---

## Task 6: 全体統合テスト（再学習 + バックテスト）

**Files:**
- 変更なし（既存パイプラインを実行）

- [ ] **Step 1: convert_boatracecsv.py を実行**

```bash
cd /home/user/test-cordex/boatrace && python convert_boatracecsv.py 2>&1 | tail -20
```

Expected:
- `直近フォームをマージ` が出力される
- `会場別フォームをマージ` が出力される
- `使用特徴量 (6X個):` で特徴量数が増加

- [ ] **Step 2: run_train55k.py を実行（Optuna + スタッキング）**

```bash
cd /home/user/test-cordex/boatrace && python run_train55k.py 2>&1 | tee /tmp/train_result.txt
```

Expected（重要チェックポイント）:
- `Optunaチューニング中 (50試行)...` × 3モデル出力される
- 以下のコマンドで `lgbm_2nd` の `best_iteration` が 3 より大きいことを確認:

```bash
python -c "import json; m=json.load(open('checkpoints/real55k/lgbm_2nd_metrics.json')); print('best_iteration:', m['best_iteration'], '  test_auc:', m['test_auc'])"
```

Expected: `best_iteration: 10以上`、`test_auc: 0.65より高い`

- [ ] **Step 3: calibrate.py を実行**

```bash
cd /home/user/test-cordex/boatrace && python calibrate.py 2>&1 | tail -20
```

- [ ] **Step 4: backtest_ev.py を実行してROIを確認**

```bash
cd /home/user/test-cordex/boatrace && python backtest_ev.py 2>&1 | grep -E "ROI|的中率|最適EV"
```

Expected: 全ベット ROI が 28.4% より改善

- [ ] **Step 5: 全テストが通ることを確認**

```bash
cd /home/user/test-cordex/boatrace && python -m pytest tests/ -v
```

Expected: すべて PASS

- [ ] **Step 6: 結果をコミット＆プッシュ**

```bash
cd /home/user/test-cordex && \
git add boatrace/checkpoints/real55k/ boatrace/logs/ boatrace/retrain_history.json 2>/dev/null; \
git add boatrace/tests/ && \
git commit -m "feat: モデル精度改善完了 — 特徴量拡充・スタッキング・Optuna" && \
git push -u origin claude/setup-boat-race-pipeline-Fq4Uz
```
