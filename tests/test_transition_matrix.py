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
    for _ in range(300):
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
