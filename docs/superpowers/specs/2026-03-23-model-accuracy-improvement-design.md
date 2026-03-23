# ボートレース予測モデル 精度改善設計書

**日付**: 2026-03-23
**目標**: 2連単 ROI 改善（現状 +28.4%）
**方針**: 特徴量拡充 + スタッキング + Optuna チューニング

---

## 現状の問題点

| モデル | Test AUC | 問題 |
|---|---|---|
| lgbm_win | 0.805 | 概ね良好 |
| lgbm_2nd | 0.656 | best_iteration=3 でほぼ未学習 |
| lgbm_top3 | 0.763 | 概ね良好 |

- `racer_recent_form.csv` / `racer_venue_form.csv` が生成済みだが特徴量未使用
- lgbm_2nd が独立予測では限界（1着との相関を無視している）
- ハイパーパラメータが全モデル共通の固定値

---

## Section 1: 特徴量拡充

### 追加特徴量グループ

**① 直近フォーム（racer_recent_form.csv）**
- `form_win_3`, `form_top2_3`, `form_avg_rank_3`, `form_avg_st_3`
- `form_win_5`, `form_top2_5`, `form_avg_rank_5`, `form_avg_st_5`
- `form_win_10`, `form_top2_10`, `form_avg_rank_10`, `form_avg_st_10`
- `form_trend_win`, `form_trend_rank`
- `form_win_streak`, `form_no_win_streak`

**② 会場別フォーム（racer_venue_form.csv）**
- `form_venue_win_5`, `form_venue_top2_5`

**③ 対戦相手特徴量（新規計算）**
- `opponent_avg_win_rate`: 対戦相手5人の平均勝率
- `opponent_max_win_rate`: 対戦相手の最強選手の勝率
- `win_rate_vs_best`: 自勝率 ÷ 最強対戦相手の勝率
- `opponent_avg_class`: 対戦相手の平均クラス
- `relative_form_rank`: 直近フォーム(form_win_5)のレース内順位

合計: 44 → 約65特徴量

### 変更ファイル
`convert_boatracecsv.py`
- `load_and_merge()`: recent_form・venue_form マージ追加
- `engineer_features()`: 対戦相手特徴量の計算追加
- `feature_cols` リストに新特徴量を追加

---

## Section 2: スタッキング（2段階予測）

### アーキテクチャ

```
[Step 1] lgbm_win を学習・予測
         全データ（train/val/test）に pred_prob_win を付与

[Step 2] 追加特徴量を生成
         - pred_prob_win: 1着予測確率
         - prob_win_max_in_race: レース内最高1着確率
         - prob_win_rank: 1着確率のレース内順位
         - is_likely_winner: 1着確率が閾値以上か（0/1）

[Step 3] lgbm_2nd を拡張特徴量で学習
         （通常64特徴量 + 上記4特徴量 = 約69特徴量）

[Step 4] lgbm_top3 を通常通り学習（スタッキングなし）
```

### 予測時の順序
1. lgbm_win で全艇の1着確率を計算
2. スタッキング特徴量を計算
3. lgbm_2nd で2着確率を計算
4. lgbm_top3 で3着以内確率を計算

### 変更ファイル
`run_train55k.py`, `predict_today.py`

---

## Section 3: Optuna ハイパーパラメータチューニング

### チューニング対象パラメータ

| パラメータ | 探索範囲 |
|---|---|
| `num_leaves` | 31 〜 255 |
| `learning_rate` | 0.005 〜 0.1 |
| `min_child_samples` | 10 〜 100 |
| `feature_fraction` | 0.5 〜 1.0 |
| `bagging_fraction` | 0.5 〜 1.0 |
| `reg_alpha` | 0.0 〜 1.0 |
| `reg_lambda` | 0.0 〜 1.0 |
| `max_depth` | 5 〜 15 |

### 実装方針
- 各モデル 50試行
- 最適化指標: val AUC 最大化
- `run_train55k.py` 内に統合
- 結果保存: `checkpoints/real55k/optuna_params.json`
- 所要時間目安: 20〜40分

---

## Section 4: 変更ファイルまとめ

| ファイル | 変更内容 |
|---|---|
| `convert_boatracecsv.py` | 特徴量追加（recent_form, venue_form, opponent features） |
| `run_train55k.py` | Optuna チューニング + スタッキング実装 |
| `predict_today.py` | スタッキング順序に合わせた予測フロー更新 |
| `monthly_retrain.py` | 変更なし |
| `calibrate.py` | 変更なし |
| `backtest_ev.py` | 変更なし |

---

## 期待効果

| 指標 | 現状 | 目標 |
|---|---|---|
| lgbm_win AUC | 0.805 | 0.82+ |
| lgbm_2nd AUC | 0.656 | 0.72〜0.74 |
| 2連単的中率 | 19.9% | 22〜24% |
| バックテスト ROI | +28.4% | +35%+ |
