# prob_exacta 条件付き確率修正 設計書

**日付**: 2026-03-22
**ステータス**: 承認済み

## 問題

現状の `prob_exacta = p1 * p2` は1着・2着の独立を仮定しているが、実際は正の相関がある。
実測では人気コンビのヒット率が4倍程度過小評価されており、EV計算が歪んでいる。

- `target_2nd = (rank == 2)` → **周辺確率** P(2着=c2) を学習
- 必要なのは **条件付き確率** P(2着=c2 | 1着=c1)

## 解決策: 遷移行列キャリブレーション

過去レース結果から P(2着=B | 1着=A) の 6×6 テーブルを計算・保存し、推論時に使用する。

### 遷移行列

- キー: `"c1-c2"` (例: `"1-2"`, `"5-3"`)、30通り
- 値: `count(1着=c1 ∧ 2着=c2) / count(1着=c1)`
- サンプル数 < 20 のセルはフォールバック `1/5 = 0.2`
- 保存先: `boatrace/checkpoints/real55k/transition_matrix.json`

### 計算式の変更

```python
# Before
prob_exacta = p1 * p2

# After
cond_p2 = trans_matrix.get(f"{c1}-{c2}", 0.2)
prob_exacta = p1 * cond_p2
```

## 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `boatrace/run_train55k.py` | 学習後にtrainデータから遷移行列を計算→保存 |
| `boatrace/backtest_ev.py` | `load_models()`で行列ロード、`predict_exacta()`で使用 |
| `boatrace/predict_today.py` | 遷移行列をロード、`prob_exacta`計算に使用 |
| `boatrace/scrape_today.py` | 同上 |

## 変更しないファイル

- `v2_*`（別パイプライン、独自の `p2_cond` ロジックあり）
- `rank_train.py`（今回スコープ外）
- `fetch_odds.py`（`prob_exacta`を受け取るだけで計算しない）

## 遷移行列の実装詳細

```python
def compute_transition_matrix(df: pd.DataFrame) -> dict:
    """
    trainデータからP(2着=c2 | 1着=c1)を計算する。
    df には 'rank', 'course', 'date', 'jyo_cd', 'race_no' が必要。
    """
    matrix = {}
    first_counts = {}
    pair_counts = {}

    for (date, jyo, race_no), g in df.groupby(["date", "jyo_cd", "race_no"]):
        first = g[g["rank"] == 1]["course"].values
        second = g[g["rank"] == 2]["course"].values
        if len(first) == 0 or len(second) == 0:
            continue
        c1, c2 = int(first[0]), int(second[0])
        first_counts[c1] = first_counts.get(c1, 0) + 1
        pair_counts[(c1, c2)] = pair_counts.get((c1, c2), 0) + 1

    MIN_SAMPLES = 20
    for (c1, c2), cnt in pair_counts.items():
        total = first_counts.get(c1, 0)
        if total >= MIN_SAMPLES:
            matrix[f"{c1}-{c2}"] = cnt / total
        else:
            matrix[f"{c1}-{c2}"] = 0.2  # フォールバック

    return matrix
```

## 期待効果

- 人気コンビ（1-2など）のEVが現実に近づく
- 荒れレースのEV過大評価が減少
- 再学習不要で即日適用可能
