# 競艇予測システム 開発ガイド (Codex / AI Coding Agent 向け)

## プロジェクト概要

ボートレース（競艇）の**2連単**（1着・2着の組み合わせ）を予測するMLシステム。
LightGBM のスタッキングモデル 3本で構成され、毎日の予測と月次の自動再学習を行う。

---

## ディレクトリ構成

```
boatrace/
├── data/raw/            # race_results_big.csv, race_payouts_big.csv
├── data/extra/          # racer_stats.csv, motor_stats.csv, jyo_course_stats.csv,
│                        # racer_recent_form.csv, racer_venue_form.csv, weather_stats.csv
├── data/processed/      # train.csv, val.csv, test.csv, feature_cols.txt
├── checkpoints/real55k/ # lgbm_win.txt, lgbm_2nd.txt, lgbm_top3.txt,
│                        # *_isotonic.pkl, thresholds.json, transition_matrix.json
├── logs/
├── download_real.py     # 公式データ取得 (.lzh解凍)
├── download_extra.py    # 選手/モーター/会場統計生成
├── convert_boatracecsv.py  # 特徴量エンジニアリング & データ分割
├── run_train55k.py      # 学習パイプライン (Optuna + LightGBM)
├── calibrate.py         # 確率キャリブレーション (Isotonic / Platt)
├── predict_today.py     # 当日予測
├── scrape_today.py      # 公式サイトのスクレイピング (レート制限付き)
├── fetch_odds.py        # リアルタイムオッズ取得
├── backtest_ev.py       # 期待値バックテスト
├── monthly_retrain.py   # 月次再学習オーケストレーション
├── notify.py            # Slack/メール通知
└── run_today.py         # 日次実行ラッパー
```

---

## モデル構成

### 学習の流れ

```
lgbm_win    : 特徴量 94本 → 1着確率 (prob_win)
lgbm_2nd    : 特徴量 94本 + スタッキング4本 → 2着確率 (prob_2nd)
lgbm_top3   : 特徴量 94本 → 3着以内確率
```

スタッキング特徴量 (lgbm_2nd への入力):
- `prob_win` — lgbm_win の出力
- `prob_win_max_in_race` — レース内他選手の最大 prob_win
- `prob_win_rank` — レース内での prob_win 順位
- `is_likely_winner` — prob_win >= 0.25 のバイナリフラグ

### データ分割

- 時系列順に Train 70% / Val 15% / Test 15%
- ウォームアップ期間（最初の 90 日）は除外（ローリング特徴量が不安定なため）
- **データリーク厳禁**: rolling は `closed='left'`、グループ化前に必ず日付ソート

---

## 主要な特徴量 (94本)

| カテゴリ | 特徴量例 |
|----------|---------|
| 選手基本 | win_rate, nirenritsu, sanrenritsu, racer_class_num |
| モーター | motor_nirenritsu, motor_rank |
| スタート | st, tenji_time, avg_st_rank |
| 会場×コース | jyo_course_win_pct, jyo_course_avg_rank |
| 交互作用 | winrate_x_course, st_x_course, motor_racer_score |
| 相対特徴量 | win_rate_rank, win_rate_vs_avg, win_rate_vs_best |
| 対戦相手 | opponent_avg_win_rate, opponent_max_win_rate, opponent_avg_class |
| フォーム | form_win_3/5/10, form_top2_3/5/10, form_trend_win, form_win_streak |
| 市場 | mkt_win_prob_90d, mkt_win_prob_180d |
| 気象 | weather_num, wind_speed, wave, water_temp |
| カレンダー | month, dayofweek, is_weekend |

---

## 定数・閾値

```python
WINNER_THRESHOLD  = 0.25   # スタッキング用「勝ちやすい選手」判定
TAKE_RATE         = 0.225  # 控除率 22.5%
WARMUP_DAYS       = 90
OPTUNA_TRIALS     = 50     # モデルごと
NUM_BOOST_ROUND   = 2000
EARLY_STOPPING    = 100
SCALE_POS_WEIGHT  ≈ 5.0    # 正例/負例比率の逆数
SCRAPE_RPS        = 4.0    # スクレイピング レート制限
```

---

## 日次・月次ワークフロー

### 日次 (09:30 JST)
```
scrape_today.py
  → 出走表・展示タイム・直前情報 取得
  → 特徴量生成 (data/extra/ をマージ)
  → predict_today.py で推論
  → predictions_YYYYMMDD.csv 出力
  → 損益分岐オッズ算出 (= 100 / prob_exacta / 100)
```

### 月次 (毎月1日 03:00 JST)
```
download_real → データ追記 → download_extra → convert_boatracecsv
  → チェックポイントバックアップ → run_train55k → calibrate
  → backtest_ev → メトリクス比較 → notify (劣化アラート)
  → retrain_history.json に追記
```

劣化アラート閾値:
- AUC 低下 > 2%
- F1 低下 > 3%
- 2連単的中率低下 > 1%

---

## コーディング規約

1. **データリーク防止**
   - `rolling(..., closed='left')` を必ず使う
   - `groupby` 前に `sort_values('date')` を実行する
   - 未来の情報を特徴量に含めない

2. **クラス不均衡**
   - `scale_pos_weight` を必ず設定する（約5.0）
   - 評価指標は AUC + F1 + Precision + Recall をセットで出す

3. **確率出力**
   - モデル出力は必ずキャリブレーション済み確率を使う
   - キャリブレーターは `calibrate.py` の Isotonic / Platt

4. **スクレイピング**
   - `scrape_today.py` のレート制限 (4 req/s) を厳守する
   - リトライは指数バックオフ (max 3回)

5. **テスト**
   - 新機能は `tests/test_features.py` に単体テストを追加する
   - 小サブセット (1000行) で動作確認してから full run する

6. **ログ**
   - `logging` モジュールを使い `logs/` 配下に出力する
   - モデルメトリクスは JSON で `checkpoints/` に保存する

---

## 現在のモデル性能 (テストセット)

| モデル | AUC | F1 | 備考 |
|--------|-----|----|------|
| lgbm_win  | ~0.78 | ~0.40 | 1着予測 |
| lgbm_2nd  | ~0.72 | ~0.35 | 2着予測 |
| 2連単的中率 | — | — | ~3.3% (ランダム基準と同程度) |

---

## よく使うコマンド

```bash
# 特徴量生成
python boatrace/convert_boatracecsv.py

# モデル学習 (Optuna + LightGBM)
python boatrace/run_train55k.py

# 確率キャリブレーション
python boatrace/calibrate.py

# 当日予測
python boatrace/scrape_today.py

# バックテスト (期待値分析)
python boatrace/backtest_ev.py

# 月次再学習
python boatrace/monthly_retrain.py
```

---

## 技術スタック

- Python 3.x
- pandas, numpy
- lightgbm
- optuna
- scikit-learn (metrics, calibration)
- joblib
- requests, BeautifulSoup4
- lhafile (.lzh 解凍)
