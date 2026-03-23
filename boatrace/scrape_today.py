#!/usr/bin/env python3
"""
scrape_today.py
ボートレース公式サイトから当日の出走表・展示情報を取得して予測を実行する。

取得データ:
  - 出走表 (racelist)  : 選手・モーター・ボート・全国/当地成績
  - 直前情報 (beforeinfo): 展示タイム・展示ST・気象

実行タイミング:
  - 各場の展示タイム発表後 (通常 09:30〜) に実行
  - 展示データがまだない場合は racelist の avg_st を代替使用
  - python scrape_today.py                → 当日全場
  - python scrape_today.py --jcd 06      → 浜名湖のみ
  - python scrape_today.py --date 20260322 → 指定日
"""

import os
import re
import sys
import time
import json
import warnings
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import requests
from datetime import datetime
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── パス定義 ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
EXTRA_DIR = os.path.join(BASE_DIR, "data", "extra")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints", "real55k")

BOATRACE_BASE = "https://www.boatrace.jp/owpc/pc/race"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ── レートリミッター ──────────────────────────────────────────────────────────

class _RateLimiter:
    """
    トークンバケット方式のレートリミッター。
    rps: 1秒あたりの最大リクエスト数。
    """
    def __init__(self, rps: float = 4.0):
        self._interval = 1.0 / rps
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_rate_limiter = _RateLimiter(rps=4.0)   # 最大 4 req/s

# スレッドセーフな print
_print_lock = threading.Lock()


def _tprint(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs)

# 場コード → 場名
VENUE_NAMES = {
    "01": "桐生", "02": "戸田",  "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津",   "10": "三国",  "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門",  "15": "丸亀",  "16": "児島",
    "17": "宮島", "18": "徳山",  "19": "下関",  "20": "若松",
    "21": "芦屋", "22": "福岡",  "23": "唐津",  "24": "大村",
}
KANJI_NUM = {"１": 1, "２": 2, "３": 3, "４": 4, "５": 5, "６": 6}


# ── HTTP ユーティリティ ───────────────────────────────────────────────────────

def _get(url: str, max_retries: int = 3, wait: float = 2.0) -> requests.Response | None:
    for attempt in range(max_retries):
        _rate_limiter.acquire()
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            _tprint(f"  HTTP {r.status_code}: {url}", file=sys.stderr)
        except requests.RequestException as e:
            _tprint(f"  Network error (attempt {attempt+1}): {e}", file=sys.stderr)
        if attempt < max_retries - 1:
            time.sleep(wait * (attempt + 1))
    return None


def _soup(url: str) -> BeautifulSoup | None:
    r = _get(url)
    if r is None:
        return None
    return BeautifulSoup(r.text, "html.parser")


# ── パースユーティリティ ─────────────────────────────────────────────────────

def _float(s: str) -> float | None:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


def _int(s: str) -> int | None:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def _parse_rates_3(text: str) -> tuple[float | None, float | None, float | None]:
    """
    '4.99 22.73 48.18' または '4.9922.7348.18' のような3値文字列を
    (win_rate, nirenritsu, sanrenritsu) に変換する。
    """
    nums = re.findall(r"\d+\.\d+", text)
    if len(nums) >= 3:
        return _float(nums[0]), _float(nums[1]), _float(nums[2])
    if len(nums) == 2:
        return _float(nums[0]), _float(nums[1]), None
    return None, None, None


def _parse_no_rate2(text: str) -> tuple[int | None, float | None, float | None]:
    """
    '56 39.91 56.58' または '5639.9156.58' を (no, niren, sanren) に変換。
    """
    parts = text.strip().split()
    if len(parts) >= 3:
        return _int(parts[0]), _float(parts[1]), _float(parts[2])
    # スペースなし版: 末尾から XX.XX を2つ取る
    nums = re.findall(r"\d+\.\d+", text)
    no_m = re.match(r"(\d+)", text)
    if nums and no_m:
        no_str = no_m.group(1)
        # noは最初の数値部分（整数のみ）
        no_end = len(no_str)
        if "." not in no_str:
            return _int(no_str), _float(nums[0]) if len(nums) > 0 else None, \
                   _float(nums[1]) if len(nums) > 1 else None
    return None, None, None


def _parse_racer_cell(text: str) -> tuple[int | None, str | None]:
    """
    '5110\n/ B1 杉山喜一…' → (racer_id=5110, racer_class='B1')
    """
    m = re.search(r"(\d{4})\s*/\s*([A-Z]\d)", text.replace("\n", " "))
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def _parse_fl_st(text: str) -> float | None:
    """
    'F0 L0 0.15' → avg_st=0.15
    """
    m = re.search(r"(\d+\.\d+)\s*$", text.strip())
    if m:
        return _float(m.group(1))
    return None


# ── 開催場取得 ────────────────────────────────────────────────────────────────

def fetch_today_venues(date_str: str) -> list[dict]:
    """
    当日の開催場リストを取得する。
    Returns: [{"jyo_cd": "06", "jyo_name": "浜名湖"}, ...]
    """
    url = f"{BOATRACE_BASE}/index?hd={date_str}"
    soup = _soup(url)
    if soup is None:
        return []

    venues = []
    seen = set()
    for a in soup.find_all("a", href=re.compile(r"jcd=\d+")):
        href = a.get("href", "")
        m = re.search(r"jcd=(\d{2})", href)
        if m and "raceindex" in href:
            jcd = m.group(1).zfill(2)
            if jcd not in seen:
                seen.add(jcd)
                venues.append({
                    "jyo_cd":   jcd,
                    "jyo_name": VENUE_NAMES.get(jcd, jcd),
                })
    return venues


def fetch_race_count(jyo_cd: str, date_str: str) -> int:
    """今日の指定場のレース数を取得する。"""
    url = f"{BOATRACE_BASE}/raceindex?jcd={jyo_cd}&hd={date_str}"
    soup = _soup(url)
    if soup is None:
        return 12  # default

    max_rno = 0
    for a in soup.find_all("a", href=re.compile(rf"rno=\d+&jcd={jyo_cd}")):
        m = re.search(r"rno=(\d+)", a.get("href", ""))
        if m:
            max_rno = max(max_rno, int(m.group(1)))
    return max_rno if max_rno > 0 else 12


# ── 出走表スクレイプ ──────────────────────────────────────────────────────────

def fetch_racelist(jyo_cd: str, race_no: int, date_str: str) -> list[dict]:
    """
    出走表から1レース分の選手データを取得する。
    Returns: 1艇 = 1 dict のリスト (最大6艇)
    """
    url = f"{BOATRACE_BASE}/racelist?rno={race_no}&jcd={jyo_cd}&hd={date_str}"
    soup = _soup(url)
    if soup is None:
        return []

    tables = soup.find_all("table")
    if len(tables) < 2:
        return []

    # コースごとに全行収集し、有効データ（racer_id あり）を優先する
    rows_by_course: dict[int, dict] = {}
    for row in tables[1].find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        waku = KANJI_NUM.get(cells[0].get_text(strip=True))
        if waku is None:
            continue

        cell_texts = [c.get_text(" ", strip=True) for c in cells]

        racer_id, racer_class = _parse_racer_cell(cell_texts[2])
        avg_st      = _parse_fl_st(cell_texts[3])
        win_rate, nirenritsu, sanrenritsu = _parse_rates_3(cell_texts[4])
        motor_no, motor_nirenritsu, _ = _parse_no_rate2(cell_texts[6])
        boat_no,  boat_nirenritsu,  _ = _parse_no_rate2(cell_texts[7])

        entry = {
            "course":          waku,
            "racer_id":        racer_id,
            "racer_class":     racer_class,
            "motor_no":        motor_no,
            "boat_no":         boat_no,
            "win_rate":        win_rate,
            "nirenritsu":      nirenritsu,
            "sanrenritsu":     sanrenritsu,
            "motor_nirenritsu": motor_nirenritsu,
            "avg_st":          avg_st,
            "tenji_time":      None,
            "st":              None,
        }
        # 有効データ（racer_id あり）を優先。未登録ならとにかく追加。
        if waku not in rows_by_course or (racer_id is not None
                                          and rows_by_course[waku]["racer_id"] is None):
            rows_by_course[waku] = entry

    return [rows_by_course[c] for c in sorted(rows_by_course)]


# ── 直前情報スクレイプ ────────────────────────────────────────────────────────

def _parse_weather(soup: BeautifulSoup) -> dict:
    """気象情報を抽出する。"""
    text = soup.get_text(" ", strip=True)

    weather_map = {"晴": 0, "曇": 1, "雨": 2, "雪": 3}
    weather_num = 0
    for kw, val in weather_map.items():
        if kw in text:
            weather_num = val
            break

    def _find(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "weather":    next((k for k in weather_map if k in text), "晴"),
        "weather_num": weather_num,
        "wind_speed":  _find(r"風速\s*([\d.]+)"),
        "wave":        _find(r"波高\s*([\d.]+)"),
        "water_temp":  _find(r"水温\s*([\d.]+)"),
    }


def fetch_beforeinfo(jyo_cd: str, race_no: int, date_str: str,
                     racer_map: dict[int, dict]) -> tuple[list[dict], dict]:
    """
    直前情報から展示タイム・展示ST・気象を取得する。

    racer_map: {course: racer_dict} (racelist から渡す)
    Returns: (展示データリスト, 気象dict)
    """
    url = f"{BOATRACE_BASE}/beforeinfo?rno={race_no}&jcd={jyo_cd}&hd={date_str}"
    soup = _soup(url)
    if soup is None:
        return [], {}

    weather = _parse_weather(soup)

    tables = soup.find_all("table")
    if len(tables) < 2:
        return [], weather

    exhibit = {}   # course → {tenji_time, st}
    rows = tables[1].find_all("tr")

    i = 0
    while i < len(rows):
        cells = rows[i].find_all(["td", "th"])
        if not cells:
            i += 1
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        course = _int(texts[0])

        if course in range(1, 7):
            tenji_time = _float(texts[4]) if len(texts) > 4 else None

            # 次の行以降にSTがあれば取得
            st = None
            for j in range(i + 1, min(i + 5, len(rows))):
                sub = rows[j].find_all(["td", "th"])
                sub_text = " ".join(c.get_text(" ", strip=True) for c in sub)
                if "ST" in sub_text:
                    m = re.search(r"ST\D*([-\d.]+)", sub_text)
                    if m:
                        try:
                            st = float(m.group(1))
                        except ValueError:
                            pass
                    break

            exhibit[course] = {"tenji_time": tenji_time, "st": st}

        i += 1

    return exhibit, weather


# ── 1レース分の並列スクレイプ ─────────────────────────────────────────────────

def fetch_one_race(jyo_cd: str, jyo_name: str,
                   race_no: int, date_str: str) -> list[dict] | None:
    """
    1レース分の出走表 + 直前情報を取得してマージした行リストを返す。
    スレッドセーフ。取得失敗時は None を返す。
    """
    racers = fetch_racelist(jyo_cd, race_no, date_str)
    if not racers:
        return None

    racer_map = {r["course"]: r for r in racers}
    exhibit, weather = fetch_beforeinfo(jyo_cd, race_no, date_str, racer_map)

    rows = []
    for racer in racers:
        c = racer["course"]
        if c in exhibit:
            ex = exhibit[c]
            if ex.get("tenji_time"):
                racer["tenji_time"] = ex["tenji_time"]
            if ex.get("st"):
                racer["st"] = ex["st"]
        if racer.get("st") is None:
            racer["st"] = racer.get("avg_st")

        racer.update({
            "date":     date_str,
            "jyo_cd":   jyo_cd,
            "jyo_name": jyo_name,
            "race_no":  race_no,
            **weather,
        })
        rows.append(racer)

    return rows


# ── 特徴量エンジニアリング ────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    scrape した生データから学習時と同じ特徴量セットを生成する。
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["month"]      = df["date"].dt.month
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["is_course1"]      = (df["course"] == 1).astype(int)
    df["is_inner_course"] = (df["course"] <= 3).astype(int)

    class_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    df["racer_class_num"] = df["racer_class"].map(class_map).fillna(2)

    df["weather_num"] = df.get("weather_num", 0)

    df["winrate_x_course"]  = df["win_rate"] * (7 - df["course"])
    df["st_x_course"]       = df["avg_st"].fillna(0.15) * df["course"]
    df["motor_racer_score"] = (
        df["motor_nirenritsu"].fillna(30) * 0.5 + df["win_rate"].fillna(5) * 10
    )

    # 追加 CSV マージ
    racer_path = os.path.join(EXTRA_DIR, "racer_stats.csv")
    if os.path.exists(racer_path):
        racer_stats = pd.read_csv(racer_path)
        # racer_stats に avg_st 等が重複する場合は scraped 版を除去して racer_stats 版を使う
        overlap = [c for c in racer_stats.columns if c in df.columns and c != "racer_id"]
        df = df.drop(columns=overlap, errors="ignore")
        df = df.merge(racer_stats, on="racer_id", how="left")

    motor_path = os.path.join(EXTRA_DIR, "motor_stats.csv")
    if os.path.exists(motor_path):
        motor_stats = pd.read_csv(motor_path)
        motor_stats["jyo_cd"] = motor_stats["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(motor_stats, on=["jyo_cd", "motor_no"], how="left")

    jyo_path = os.path.join(EXTRA_DIR, "jyo_course_stats.csv")
    if os.path.exists(jyo_path):
        jyo_stats = pd.read_csv(jyo_path)
        jyo_stats["jyo_cd"] = jyo_stats["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(
            jyo_stats[["jyo_cd", "course", "jyo_course_win_pct", "jyo_course_avg_rank"]],
            on=["jyo_cd", "course"], how="left",
        )

    # 市場払戻特徴量 (payout stats から)
    payout_stats_path = os.path.join(EXTRA_DIR, "jyo_course_payout_stats.csv")
    if os.path.exists(payout_stats_path):
        pay_stats = pd.read_csv(payout_stats_path)
        pay_stats["jyo_cd"] = pay_stats["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(pay_stats, on=["jyo_cd", "course"], how="left")

    # 直近フォーム特徴量
    form_path = os.path.join(EXTRA_DIR, "racer_recent_form.csv")
    if os.path.exists(form_path):
        form_df = pd.read_csv(form_path)
        df = df.merge(form_df, on="racer_id", how="left")

    venue_form_path = os.path.join(EXTRA_DIR, "racer_venue_form.csv")
    if os.path.exists(venue_form_path):
        venue_form = pd.read_csv(venue_form_path)
        venue_form["jyo_cd"] = venue_form["jyo_cd"].astype(str).str.zfill(2)
        df = df.merge(venue_form, on=["racer_id", "jyo_cd"], how="left")

    # レース内相対特徴量
    race_group = ["date", "jyo_cd", "race_no"]
    for col, asc, out in [
        ("win_rate",        False, "win_rate_rank"),
        ("st",              True,  "st_rank"),
        ("tenji_time",      True,  "tenji_rank"),
        ("nirenritsu",      False, "nirenritsu_rank"),
        ("motor_nirenritsu",False, "motor_rank"),
        ("avg_st",          True,  "avg_st_rank"),
    ]:
        if col in df.columns:
            df[out] = df.groupby(race_group)[col].rank(ascending=asc)

    for col, out in [
        ("win_rate",   "win_rate_vs_avg"),
        ("st",         "st_vs_avg"),
        ("tenji_time", "tenji_vs_avg"),
    ]:
        if col in df.columns:
            df[out] = df[col] - df.groupby(race_group)[col].transform("mean")

    # 欠損値をゼロ/中央値で補完
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(df[numeric].median())

    return df


# ── 予測 ─────────────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """モデル・閾値・キャリブレーターをロードする。"""
    model_win = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_win.txt"))
    model_2nd = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_2nd.txt"))

    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    thr_path = os.path.join(CKPT_DIR, "thresholds.json")
    thresholds = {"lgbm_win": 0.5, "lgbm_2nd": 0.5}
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            thresholds = json.load(f)

    calibrators = {}
    for mname in ("lgbm_win", "lgbm_2nd"):
        for method in ("isotonic", "platt"):
            p = os.path.join(CKPT_DIR, f"{mname}_{method}.pkl")
            if os.path.exists(p):
                calibrators[f"{mname}_{method}"] = joblib.load(p)
                break  # isotonic 優先で1つだけ読む

    # 遷移行列ロード
    tm_path = os.path.join(CKPT_DIR, "transition_matrix.json")
    trans_matrix = {}
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            trans_matrix = json.load(f)

    return model_win, model_2nd, feature_cols, thresholds, calibrators, trans_matrix


def _apply_cal(raw: np.ndarray, calibrators: dict, key: str) -> np.ndarray:
    if key not in calibrators:
        return raw
    cal = calibrators[key]
    if hasattr(cal, "predict_proba"):
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return np.clip(cal.predict(raw), 1e-7, 1 - 1e-7)


def predict_races(df: pd.DataFrame, model_win, model_2nd,
                  feature_cols: list, thresholds: dict,
                  calibrators: dict, trans_matrix: dict = None) -> tuple:
    """
    全レースの2連単予測を返す。

    Returns:
        (exacta_df, boat_df) のタプル。
        exacta_df: 1レース1行 (後方互換)
        boat_df:   1艇1行の艇別確率 (EV選択用)
    """
    if trans_matrix is None:
        trans_matrix = {}

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  警告: 特徴量不足 {len(missing)}個 → 0で補完: {missing[:5]}...")
        for c in missing:
            df[c] = 0.0

    X = df[feature_cols].values

    raw_win = model_win.predict(X, num_iteration=model_win.best_iteration)
    raw_2nd = model_2nd.predict(X, num_iteration=model_2nd.best_iteration)

    df = df.copy()
    df["prob_win"] = _apply_cal(raw_win, calibrators, "lgbm_win_isotonic")
    df["prob_2nd"] = _apply_cal(raw_2nd, calibrators, "lgbm_2nd_isotonic")

    # ── 艇別確率 DataFrame (EV選択で使用) ────────────────────────────────────
    boat_rows = []
    for _, row in df.iterrows():
        boat_rows.append({
            "jyo_cd":   row["jyo_cd"],
            "jyo_name": row["jyo_name"],
            "race_no":  int(row["race_no"]),
            "course":   int(row["course"]),
            "prob_win": round(float(row["prob_win"]), 4),
            "prob_2nd": round(float(row["prob_2nd"]), 4),
        })
    boat_df = pd.DataFrame(boat_rows)

    # ── レース単位の予測 DataFrame (後方互換) ────────────────────────────────
    rows = []
    for (jyo_cd, jyo_name, race_no), g in df.groupby(
            ["jyo_cd", "jyo_name", "race_no"]):
        idx1 = g["prob_win"].idxmax()
        pred_1st = g.loc[idx1, "course"]
        p1 = g.loc[idx1, "prob_win"]

        rest = g[g.index != idx1]
        idx2 = rest["prob_2nd"].idxmax()
        pred_2nd = rest.loc[idx2, "course"]
        p2 = rest.loc[idx2, "prob_2nd"]

        cond_p2 = trans_matrix.get(f"{int(pred_1st)}-{int(pred_2nd)}", p2)
        prob_exacta = p1 * cond_p2
        breakeven = round(100 / max(prob_exacta, 1e-6) / 100) * 100

        rows.append({
            "jyo_cd":       jyo_cd,
            "jyo_name":     jyo_name,
            "race_no":      int(race_no),
            "pred_1st":     int(pred_1st),
            "pred_2nd":     int(pred_2nd),
            "prob_win":     round(p1, 4),
            "prob_2nd":     round(p2, 4),
            "prob_exacta":  round(prob_exacta, 4),
            "breakeven_odds": int(breakeven),
        })

    return pd.DataFrame(rows), boat_df


# ── 表示 ─────────────────────────────────────────────────────────────────────

def print_predictions(pred_df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("2連単 予測結果")
    print("=" * 78)

    for (jyo_cd, jyo_name), jyo_df in pred_df.groupby(["jyo_cd", "jyo_name"]):
        print(f"\n{'━' * 78}")
        print(f"  {jyo_name} ({jyo_cd})")
        print(f"{'━' * 78}")
        print(f"  {'R':>3}  {'予測':^10}  {'1着%':>6}  {'2着%':>6}  {'損益分岐オッズ':>14}")
        for _, row in jyo_df.sort_values("race_no").iterrows():
            print(
                f"  {int(row['race_no']):2d}R  "
                f"【{row['pred_1st']}-{row['pred_2nd']}】        "
                f"{row['prob_win']:>5.1%}  {row['prob_2nd']:>5.1%}  "
                f"  {row['breakeven_odds']:>6}円以上なら買い"
            )

    print(f"\n{'=' * 78}")
    print("注目 2連単 (損益分岐オッズが低い順 = モデル確信度が高い)")
    print(f"{'=' * 78}")
    print(f"  {'場':>6} {'R':>3}  {'組合せ':^10}  {'2連単確率':>10}  {'損益分岐':>10}")
    top = pred_df.sort_values("breakeven_odds").head(10)
    for _, row in top.iterrows():
        print(
            f"  {row['jyo_name']:>6} {int(row['race_no']):>3}R  "
            f"【{row['pred_1st']}-{row['pred_2nd']}】        "
            f"{row['prob_exacta']:>9.1%}  {row['breakeven_odds']:>8}円"
        )

    print("\n※ 実際のオッズが「損益分岐オッズ」を超えていれば期待値プラスの買い目。")


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ボートレース当日予測スクレイパー")
    parser.add_argument("--date",  default=datetime.now().strftime("%Y%m%d"),
                        help="対象日 (YYYYMMDD, default: 今日)")
    parser.add_argument("--jcd",   default=None,
                        help="場コード (例: 06 = 浜名湖)。省略時は全場")
    parser.add_argument("--races", default=None,
                        help="レース番号 (例: 1-6 または 1,3,5)。省略時は全レース")
    parser.add_argument("--out",     default=None,
                        help="予測結果 CSV の保存先")
    parser.add_argument("--workers", type=int, default=6,
                        help="並列スクレイプ数 (default: 6)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"ボートレース当日予測  {args.date}")
    print("=" * 60)

    # ── 開催場の決定 ─────────────────────────────────────────────────────────
    if args.jcd:
        venues = [{"jyo_cd": args.jcd.zfill(2),
                   "jyo_name": VENUE_NAMES.get(args.jcd.zfill(2), args.jcd)}]
    else:
        print("開催場を取得中...")
        venues = fetch_today_venues(args.date)
        if not venues:
            print("本日の開催場が見つかりません。")
            return
        print(f"  {len(venues)}場: " + ", ".join(v["jyo_name"] for v in venues))

    # ── レース番号フィルタ ────────────────────────────────────────────────────
    race_filter = None
    if args.races:
        if "-" in args.races:
            lo, hi = args.races.split("-")
            race_filter = set(range(int(lo), int(hi) + 1))
        else:
            race_filter = {int(x) for x in args.races.split(",")}

    # ── モデルロード ──────────────────────────────────────────────────────────
    print("モデルロード中...")
    model_win, model_2nd, feature_cols, thresholds, calibrators, trans_matrix = load_models()
    if calibrators:
        print(f"  キャリブレーター: {list(calibrators.keys())}")

    # ── スクレイプ (並列) ─────────────────────────────────────────────────────
    # レースカウントは軽量なので先に逐次取得
    jobs: list[tuple[str, str, int]] = []
    for venue in venues:
        jyo_cd   = venue["jyo_cd"]
        jyo_name = venue["jyo_name"]
        n_races  = fetch_race_count(jyo_cd, args.date)
        print(f"  {jyo_name}({jyo_cd}) {n_races}レース")
        for race_no in range(1, n_races + 1):
            if race_filter and race_no not in race_filter:
                continue
            jobs.append((jyo_cd, jyo_name, race_no))

    print(f"\n{len(jobs)}レース を {args.workers} workers で並列取得中...")

    all_rows: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(fetch_one_race, jyo_cd, jyo_name, race_no, args.date):
                (jyo_cd, jyo_name, race_no)
            for jyo_cd, jyo_name, race_no in jobs
        }
        for future in as_completed(future_to_job):
            jyo_cd, jyo_name, race_no = future_to_job[future]
            done += 1
            try:
                rows = future.result()
            except Exception as e:
                _tprint(f"  [{done}/{len(jobs)}] {jyo_name} {race_no}R ERROR: {e}",
                        file=sys.stderr)
                continue

            if rows is None:
                _tprint(f"  [{done}/{len(jobs)}] {jyo_name} {race_no}R データなし")
                continue

            tenji_ok = sum(1 for r in rows if r.get("tenji_time") is not None)
            _tprint(f"  [{done}/{len(jobs)}] {jyo_name} {race_no}R OK "
                    f"({tenji_ok}/6 展示タイム)")
            all_rows.extend(rows)

    if not all_rows:
        print("\nデータが取得できませんでした。")
        return

    # ── 特徴量生成・予測 ─────────────────────────────────────────────────────
    print("\n特徴量生成中...")
    df = pd.DataFrame(all_rows)
    df = prepare_features(df)

    print("予測中...")
    pred_df, boat_df = predict_races(df, model_win, model_2nd,
                                     feature_cols, thresholds, calibrators,
                                     trans_matrix=trans_matrix)

    print_predictions(pred_df)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out_path = args.out or os.path.join(
        CKPT_DIR, f"predictions_{args.date}.csv"
    )
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 艇別確率 CSV (fetch_odds.py の EV 選択モードで使用)
    boat_out = out_path.replace("predictions_", "boat_probs_")
    boat_df.to_csv(boat_out, index=False, encoding="utf-8-sig")

    print(f"\n予測結果保存: {out_path}")
    print(f"艇別確率保存: {boat_out}")
    print("完了!")


if __name__ == "__main__":
    main()
