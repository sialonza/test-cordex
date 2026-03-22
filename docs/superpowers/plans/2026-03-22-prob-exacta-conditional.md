# prob_exacta 条件付き確率修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `prob_exacta = p1 * p2`（独立仮定）を `prob_win[c1] × P(2着=c2 | 1着=c1)` に修正し、EV計算の過小評価を解消する。

**Architecture:** 学習時に trainデータから遷移行列 `transition_matrix.json` を計算・保存し、バックテスト・予測スクリプトがこのファイルをロードして条件付き確率として使う。モデルの再学習は不要。

**Tech Stack:** Python 3, pandas, json, lightgbm (既存), pytest

---

## File Map

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `boatrace/run_train55k.py` | 修正 | 学習後に `compute_transition_matrix()` を呼び出し `transition_matrix.json` 保存 |
| `boatrace/backtest_ev.py` | 修正 | `load_models()` で行列ロード、`predict_exacta()` で条件付き確率を使用 |
| `boatrace/predict_today.py` | 修正 | `main()` で行列ロード、`prob_exacta` 計算に使用 |
| `boatrace/scrape_today.py` | 修正 | `predict_races()` で行列ロード・使用 |
| `tests/test_transition_matrix.py` | 新規作成 | `compute_transition_matrix()` のユニットテスト |

---

### Task 1: `compute_transition_matrix()` のテストを書く

**Files:**
- Create: `tests/test_transition_matrix.py`

- [ ] **Step 1: テストファイルを作成する**

```python
# tests/test_transition_matrix.py
import pandas as pd
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def compute_transition_matrix(df):
    """run_train55k.py からインポートする前のスタブ — 後で差し替える。"""
    from boatrace.run_train55k import compute_transition_matrix as _fn
    return _fn(df)


def make_df(pairs):
    """(c1, c2) のリストからテスト用 DataFrame を作る。"""
    rows = []
    for i, (c1, c2) in enumerate(pairs):
        # 1着ボート
        rows.append({"date": "2024-01-01", "jyo_cd": "06",
                     "race_no": i + 1, "rank": 1, "course": c1})
        # 2着ボート
        rows.append({"date": "2024-01-01", "jyo_cd": "06",
                     "race_no": i + 1, "rank": 2, "course": c2})
        # 残り艇 (rank3-6, 他のコース)
        used = {c1, c2}
        rank = 3
        for course in range(1, 7):
            if course not in used:
                rows.append({"date": "2024-01-01", "jyo_cd": "06",
                             "race_no": i + 1, "rank": rank, "course": course})
                rank += 1
    return pd.DataFrame(rows)


def test_basic_probability():
    """コース1が1着の30レース中20回コース2が2着 → P(2|1) = 20/30 ≈ 0.667"""
    pairs = [("1", "2")] * 20 + [("1", "3")] * 10
    df = make_df(pairs)
    matrix = compute_transition_matrix(df)
    assert abs(matrix["1-2"] - 20 / 30) < 1e-9
    assert abs(matrix["1-3"] - 10 / 30) < 1e-9


def test_all_probs_sum_to_one_per_first():
    """1着=Aの条件下での全2着確率の合計 ≈ 1.0"""
    import random
    random.seed(42)
    courses = list(range(1, 7))
    pairs = []
    for _ in range(100):
        c1 = random.choice(courses)
        c2 = random.choice([c for c in courses if c != c1])
        pairs.append((str(c1), str(c2)))

    df = make_df(pairs)
    matrix = compute_transition_matrix(df)

    for first in range(1, 7):
        total = sum(matrix.get(f"{first}-{s}", 0.0)
                    for s in range(1, 7) if s != first)
        # 計算されたキーの総和が 1.0 に近いこと (フォールバック除く)
        if total > 0:
            assert abs(total - 1.0) < 1e-6, f"first={first}, total={total}"


def test_fallback_for_sparse_cells():
    """サンプル数が MIN_SAMPLES(20) 未満のセルはフォールバック 0.2 を返す。"""
    # コース1が1着・コース2が2着のレースを5回だけ用意
    pairs = [("1", "2")] * 5
    df = make_df(pairs)
    matrix = compute_transition_matrix(df)
    assert matrix.get("1-2") == pytest.approx(0.2)


def test_keys_are_string_hyphen_format():
    """キー形式は '1-2' のような文字列であること。"""
    pairs = [("1", "2")] * 30
    df = make_df(pairs)
    matrix = compute_transition_matrix(df)
    assert "1-2" in matrix
    assert isinstance(list(matrix.keys())[0], str)
    assert "-" in list(matrix.keys())[0]
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd /home/user/test-cordex && python -m pytest tests/test_transition_matrix.py -v 2>&1 | head -30
```

期待: `ImportError` または `AttributeError`（関数未実装のため）

---

### Task 2: `compute_transition_matrix()` を `run_train55k.py` に実装する

**Files:**
- Modify: `boatrace/run_train55k.py:218`（`main()` の直前に関数を挿入）

- [ ] **Step 1: 関数を `main()` の直前に追加する**

`boatrace/run_train55k.py` の `def main():` の直前（217行目付近）に以下を挿入:

```python
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
```

- [ ] **Step 2: `main()` 内の閾値保存ブロックの直後に行列の計算・保存を追加する**

`run_train55k.py` の以下の行（`print(f"\nチェックポイント: {CKPT_DIR}")` の直前）に追加:

```python
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
```

> **注意**: `all_train = pd.concat([train, val])` を使う。test データは含めない（リークを防ぐため）。

- [ ] **Step 3: テストを実行して全パスを確認する**

```bash
cd /home/user/test-cordex && python -m pytest tests/test_transition_matrix.py -v
```

期待: 全テスト PASSED

- [ ] **Step 4: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/run_train55k.py tests/test_transition_matrix.py && git commit -m "feat: add compute_transition_matrix() for conditional exacta probability"
```

---

### Task 3: `backtest_ev.py` を遷移行列対応に修正する

**Files:**
- Modify: `boatrace/backtest_ev.py`

- [ ] **Step 1: `load_models()` に遷移行列のロードを追加する**

`load_models()` の `return` 文の直前（51行目付近）を以下のように変更:

```python
    # 遷移行列 (存在する場合のみ読み込む)
    tm_path = os.path.join(CKPT_DIR, "transition_matrix.json")
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            trans_matrix = json.load(f)
        print(f"遷移行列読み込み: {len(trans_matrix)}セル")
    else:
        trans_matrix = {}
        print("警告: transition_matrix.json が見つかりません。独立近似を使用します。")

    return model_win, model_2nd, feature_cols, thresholds, calibrators, trans_matrix
```

- [ ] **Step 2: `predict_exacta()` のシグネチャと呼び出し元を更新する**

`predict_exacta()` の引数に `trans_matrix=None` を追加（71行目付近）:

```python
def predict_exacta(test_df, model_win, model_2nd, feature_cols,
                   thresholds=None, calibrators=None, trans_matrix=None):
```

関数冒頭の `calibrators = {}` の直後に追加:

```python
    if trans_matrix is None:
        trans_matrix = {}
```

- [ ] **Step 3: `prob_exacta` の計算行を修正する（128行目付近）**

変更前:
```python
            "prob_exacta": p1 * p2,   # 2連単確率の近似
```

変更後:
```python
            "prob_exacta": p1 * trans_matrix.get(f"{int(c1)}-{int(c2)}", 0.2),
```

> `trans_matrix` が空のとき（ファイルなし）はフォールバック `0.2`（均等分布 1/5）を使用する。`p2`（旧モデルの周辺確率）に戻してしまうと独立仮定のバイアスが再発するため使わない。

- [ ] **Step 4: `main()` の `load_models()` と `predict_exacta()` 呼び出しを更新する**

変更前（262〜270行目付近）:
```python
    model_win, model_2nd, feature_cols, thresholds, calibrators = load_models()
    ...
    pred_df = predict_exacta(test, model_win, model_2nd, feature_cols,
                             thresholds, calibrators)
```

変更後:
```python
    model_win, model_2nd, feature_cols, thresholds, calibrators, trans_matrix = load_models()
    ...
    pred_df = predict_exacta(test, model_win, model_2nd, feature_cols,
                             thresholds, calibrators, trans_matrix)
```

- [ ] **Step 5: 構文チェック**

```bash
cd /home/user/test-cordex && python -c "import boatrace.backtest_ev; print('OK')"
```

期待: `OK`

- [ ] **Step 6: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/backtest_ev.py && git commit -m "feat: use transition matrix for prob_exacta in backtest_ev"
```

---

### Task 4: `predict_today.py` を遷移行列対応に修正する

**Files:**
- Modify: `boatrace/predict_today.py`

- [ ] **Step 1: `main()` のモデルロード後に遷移行列ロードを追加する**

`main()` 内の `model_2nd = lgb.Booster(...)` の直後（178行目付近）に追加:

```python
    # 遷移行列ロード
    tm_path = os.path.join(CKPT_DIR, "transition_matrix.json")
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            trans_matrix = json.load(f)
    else:
        trans_matrix = {}
```

- [ ] **Step 2: `prob_exacta` の計算行を修正する（211行目付近）**

変更前:
```python
        prob_exacta = pred_1st["prob_win"] * pred_2nd["prob_2nd"]
```

変更後:
```python
        c1_key = f"{int(pred_1st['course'])}-{int(pred_2nd['course'])}"
        cond_p2 = trans_matrix.get(c1_key, pred_2nd["prob_2nd"])
        prob_exacta = pred_1st["prob_win"] * cond_p2
```

- [ ] **Step 3: 構文チェック**

```bash
cd /home/user/test-cordex && python -c "import boatrace.predict_today; print('OK')"
```

期待: `OK`

- [ ] **Step 4: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/predict_today.py && git commit -m "feat: use transition matrix for prob_exacta in predict_today"
```

---

### Task 5: `scrape_today.py` を遷移行列対応に修正する

**Files:**
- Modify: `boatrace/scrape_today.py`

- [ ] **Step 1: `predict_races()` のシグネチャに `trans_matrix` を追加する（535行目付近）**

変更前:
```python
def predict_races(df: pd.DataFrame, model_win, model_2nd,
                  feature_cols: list, thresholds: dict,
                  calibrators: dict) -> pd.DataFrame:
```

変更後:
```python
def predict_races(df: pd.DataFrame, model_win, model_2nd,
                  feature_cols: list, thresholds: dict,
                  calibrators: dict, trans_matrix: dict = None) -> pd.DataFrame:
```

関数冒頭の `rows = []` の直前に追加:

```python
    if trans_matrix is None:
        trans_matrix = {}
```

- [ ] **Step 2: `prob_exacta` の計算行を修正する（566行目付近）**

変更前:
```python
        prob_exacta = p1 * p2
```

変更後:
```python
        cond_p2 = trans_matrix.get(f"{int(pred_1st)}-{int(pred_2nd)}", p2)
        prob_exacta = p1 * cond_p2
```

- [ ] **Step 3: `scrape_today.py` の `load_models()` に遷移行列ロードを追加し、呼び出し元に渡す**

`scrape_today.py` の `load_models()` 関数（501行目付近）の `return` 文の直前に追加:

```python
    # 遷移行列ロード
    tm_path = os.path.join(CKPT_DIR, "transition_matrix.json")
    trans_matrix = {}
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            trans_matrix = json.load(f)

    return model_win, model_2nd, feature_cols, thresholds, calibrators, trans_matrix
```

`main()` 内の `load_models()` 呼び出し（`scrape_today.py` の `main()` 内）を変更:

変更前:
```python
    model_win, model_2nd, feature_cols, thresholds, calibrators = load_models(...)
    ...
    pred_df = predict_races(df, model_win, model_2nd,
                            feature_cols, thresholds, calibrators)
```

変更後:
```python
    model_win, model_2nd, feature_cols, thresholds, calibrators, trans_matrix = load_models(...)
    ...
    pred_df = predict_races(df, model_win, model_2nd,
                            feature_cols, thresholds, calibrators, trans_matrix=trans_matrix)
```

> `load_models()` の呼び出し行は `grep -n "load_models()" boatrace/scrape_today.py` で確認する。

- [ ] **Step 4: 構文チェック**

```bash
cd /home/user/test-cordex && python -c "import boatrace.scrape_today; print('OK')"
```

期待: `OK`

- [ ] **Step 5: コミット**

```bash
cd /home/user/test-cordex && git add boatrace/scrape_today.py && git commit -m "feat: use transition matrix for prob_exacta in scrape_today"
```

---

### Task 6: 遷移行列を実際のデータで生成し動作確認する

**Files:**
- 生成: `boatrace/checkpoints/real55k/transition_matrix.json`

- [ ] **Step 1: 既存の学習済みモデルで遷移行列だけを生成するスタンドアロンスクリプトを実行する**

> `run_train55k.py` をフル再実行すると時間がかかるため、以下の1ライナーで trainデータから行列だけ生成できる:

```bash
cd /home/user/test-cordex && python - <<'EOF'
import json, os, sys
sys.path.insert(0, ".")
import pandas as pd
from boatrace.run_train55k import compute_transition_matrix, CKPT_DIR, DATA_DIR

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
all_train = pd.concat([train, val], ignore_index=True)
tm = compute_transition_matrix(all_train)
out = os.path.join(CKPT_DIR, "transition_matrix.json")
with open(out, "w") as f:
    json.dump(tm, f, indent=2)
print(f"保存: {out}  ({len(tm)}セル)")
for k in ["1-2","1-3","2-1","3-1","6-5"]:
    print(f"  {k}: {tm.get(k, 'N/A'):.3f}")
EOF
```

期待: 30セル分の確率が出力され、`transition_matrix.json` が生成される。

- [ ] **Step 2: backtest_ev.py を実行して EV計算が変化したか確認する**

```bash
cd /home/user/test-cordex && python -m boatrace.backtest_ev 2>&1 | head -40
```

期待:
- `遷移行列読み込み: 30セル` が出力される
- `prob_exacta` が以前より高い値になる（人気コンビで顕著）

- [ ] **Step 3: 全テストを実行する**

```bash
cd /home/user/test-cordex && python -m pytest tests/ -v
```

期待: 全 PASSED

- [ ] **Step 4: 最終コミット**

```bash
cd /home/user/test-cordex && git add boatrace/checkpoints/real55k/transition_matrix.json && git commit -m "data: add transition_matrix.json computed from train+val data"
```

---

## 検証チェックリスト

修正後に確認すること:

- [ ] `transition_matrix.json` が30キーを持つ（1〜6コース、自己ループ除く全ペア）
- [ ] 各 `c1` に対する確率の和 ≈ 1.0（例: `1-2 + 1-3 + 1-4 + 1-5 + 1-6 ≈ 1.0`）
- [ ] `backtest_ev.py` の出力に `遷移行列読み込み` が含まれる
- [ ] `prob_exacta` の中央値が修正前より上昇している（特に1-2, 1-3などの人気コンビ）
- [ ] `trans_matrix.json` が存在しない環境でも各スクリプトが従来通り動作する（フォールバック確認）
