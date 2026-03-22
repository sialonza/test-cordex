#!/usr/bin/env python3
"""
fetch_odds.py
2連単リアルタイムオッズを公式サイトから取得し、モデル予測と照合して EV を計算する。

EV (期待値) = モデル予測確率 × 実際のオッズ倍率
  EV > 1.0 → プラス収支期待  (= 実際オッズ > 損益分岐オッズ)
  EV < 1.0 → マイナス収支期待 (見送り推奨)

使い方:
  # scrape_today.py の予測CSVと組み合わせてEV計算 (最も典型的な使い方)
  python fetch_odds.py --pred checkpoints/real55k/predictions_20260322.csv

  # 場・レースを直接指定してオッズ表を表示 (予測なし)
  python fetch_odds.py --jcd 06 --date 20260322
  python fetch_odds.py --jcd 06 --races 5-8

  # 予測CSVなしで全30組み合わせのオッズ一覧を表示
  python fetch_odds.py --jcd 06 --date 20260322 --all

ワークフロー:
  python scrape_today.py --jcd 06           # 出走表スクレイプ + モデル予測
  python fetch_odds.py --pred checkpoints/real55k/predictions_20260322.csv
"""

import os
import re
import sys
import time
import argparse
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── パス定義 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "real55k")

BOATRACE_BASE = "https://www.boatrace.jp/owpc/pc/race"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

VENUE_NAMES = {
    "01": "桐生", "02": "戸田",  "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡",  "08": "常滑",
    "09": "津",   "10": "三国",  "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門",  "15": "丸亀",  "16": "児島",
    "17": "宮島", "18": "徳山",  "19": "下関",  "20": "若松",
    "21": "芦屋", "22": "福岡",  "23": "唐津",  "24": "大村",
}

# 2連単の全組み合わせ (30通り): (1着コース, 2着コース)
ALL_EXACTA = [(c1, c2) for c1 in range(1, 7) for c2 in range(1, 7) if c1 != c2]


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _get(url: str, max_retries: int = 3, wait: float = 2.0):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            print(f"  HTTP {r.status_code}: {url}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  Network error (attempt {attempt+1}): {e}", file=sys.stderr)
        if attempt < max_retries - 1:
            time.sleep(wait * (attempt + 1))
    return None


def _soup(url: str):
    r = _get(url)
    return BeautifulSoup(r.text, "html.parser") if r else None


# ── オッズ取得 ────────────────────────────────────────────────────────────────

def fetch_odds_exacta(jyo_cd: str, race_no: int, date_str: str) -> dict[tuple, float]:
    """
    2連単オッズを取得する。

    Returns:
        {(1着コース, 2着コース): 倍率}  例: {(1, 2): 8.7, (1, 3): 11.7, ...}
        倍率 = yen_payout / 100  (8.7 なら 100円ベットで870円払戻)
        0.0 = オッズ未発売 or 出走取消
    """
    url = f"{BOATRACE_BASE}/odds2tf?rno={race_no}&jcd={jyo_cd}&hd={date_str}"
    soup = _soup(url)
    if soup is None:
        return {}

    tables = soup.find_all("table")
    # table[1] が 2連単オッズテーブル (60 td: 30コースペア + 30オッズ)
    if len(tables) < 2:
        return {}

    t = tables[1]
    rows = t.find_all("tr")
    if len(rows) < 2:
        return {}

    # ── ヘッダー行から1着コースのリストを取得 ────────────────────────────────
    header_cells = rows[0].find_all(["td", "th"])
    c1_list = []
    for cell in header_cells:
        text = cell.get_text(strip=True)
        if text.isdigit() and 1 <= int(text) <= 6:
            c1_list.append(int(text))

    if not c1_list:
        # フォールバック: 1-6 順と仮定
        c1_list = list(range(1, 7))

    # ── データ行から (2着コース, オッズ) を読む ──────────────────────────────
    # 各行: 6列 × [2着コース, オッズ] = 12セル
    odds: dict[tuple, float] = {}
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        for col_idx, c1 in enumerate(c1_list):
            base = col_idx * 2
            if base + 1 >= len(cells):
                continue
            c2_text   = cells[base].get_text(strip=True)
            odds_text = cells[base + 1].get_text(strip=True).replace(",", "")
            if not c2_text.isdigit():
                continue
            c2 = int(c2_text)
            try:
                multiplier = float(odds_text)
            except ValueError:
                multiplier = 0.0
            odds[(c1, c2)] = multiplier

    return odds


def fetch_odds_for_races(jyo_cd: str, race_nos: list[int],
                          date_str: str, verbose: bool = True) -> dict[int, dict]:
    """複数レースのオッズを一括取得する。"""
    result = {}
    for rno in race_nos:
        if verbose:
            print(f"  {rno}R オッズ取得中...", end=" ", flush=True)
        odds = fetch_odds_exacta(jyo_cd, rno, date_str)
        result[rno] = odds
        if verbose:
            n_valid = sum(1 for v in odds.values() if v > 0)
            print(f"OK ({n_valid}/30 有効)")
        time.sleep(0.5)  # サーバー負荷軽減
    return result


# ── EV 計算 ───────────────────────────────────────────────────────────────────

def compute_ev(prob_exacta: float, multiplier: float) -> float:
    """EV = モデル予測確率 × 実際のオッズ倍率。1.0 超でプラス期待値。"""
    return prob_exacta * multiplier


def ev_for_predictions(pred_df: pd.DataFrame,
                        odds_by_race: dict[int, dict]) -> pd.DataFrame:
    """
    predictions DataFrame に actual_odds / ev / bet 列を追加する。

    pred_df 列: jyo_cd, jyo_name, race_no, pred_1st, pred_2nd,
                prob_win, prob_2nd, prob_exacta, breakeven_odds
    """
    pred_df = pred_df.copy()
    actual_odds_col = []
    ev_col = []

    for _, row in pred_df.iterrows():
        rno  = int(row["race_no"])
        c1   = int(row["pred_1st"])
        c2   = int(row["pred_2nd"])
        odds = odds_by_race.get(rno, {}).get((c1, c2), None)

        if odds is None or odds == 0.0:
            actual_odds_col.append(None)
            ev_col.append(None)
        else:
            actual_odds_col.append(odds)
            ev_col.append(compute_ev(float(row["prob_exacta"]), odds))

    pred_df["actual_odds"] = actual_odds_col
    pred_df["ev"]          = ev_col
    pred_df["bet"]         = pred_df["ev"].apply(
        lambda v: "◎ BET" if v is not None and v >= 1.0 else
                  ("△ 様子見" if v is not None and v >= 0.8 else "")
    )
    return pred_df


def all_combos_ev(race_no: int, odds_dict: dict[tuple, float],
                  prob_win: dict[int, float],
                  prob_2nd: dict[int, float]) -> pd.DataFrame:
    """
    全30組み合わせの EV テーブルを返す。

    prob_win / prob_2nd は {コース番号: 確率} の形式。
    EV の近似: prob_exacta ≈ prob_win[c1] × prob_2nd[c2] / (1 - prob_win[c1])
    (1着確定後に残り5艇から2着を選ぶ条件付き確率の近似)
    """
    rows = []
    for c1, c2 in ALL_EXACTA:
        odds = odds_dict.get((c1, c2), 0.0)
        if odds <= 0:
            continue
        p1 = prob_win.get(c1, 0.0)
        p2 = prob_2nd.get(c2, 0.0)
        # 条件付き確率の近似
        denom = max(1.0 - p1, 1e-6)
        prob_approx = p1 * (p2 / denom)
        ev = compute_ev(prob_approx, odds)
        rows.append({
            "race_no": race_no,
            "combo":   f"{c1}-{c2}",
            "c1": c1, "c2": c2,
            "odds":    odds,
            "prob_approx": round(prob_approx, 4),
            "ev":      round(ev, 3),
        })
    return pd.DataFrame(rows).sort_values("ev", ascending=False)


# ── 表示 ─────────────────────────────────────────────────────────────────────

def print_odds_table(race_no: int, odds_dict: dict[tuple, float],
                     jyo_name: str = "") -> None:
    """2連単オッズを 6×5 マトリクスで表示する。"""
    header = f"  {jyo_name} {race_no}R  2連単オッズ"
    print(f"\n{header}")
    print(f"  {'':5s}", end="")
    for c2 in range(1, 7):
        print(f"  2着{c2}", end="")
    print()
    print("  " + "─" * 40)
    for c1 in range(1, 7):
        print(f"  1着{c1}", end="")
        for c2 in range(1, 7):
            if c1 == c2:
                print(f"  {'─':>4}", end="")
            else:
                v = odds_dict.get((c1, c2), 0.0)
                print(f"  {v:>4.1f}", end="")
        print()


def print_ev_summary(pred_df: pd.DataFrame) -> None:
    """EV 付き予測一覧を表示する。"""
    print("\n" + "=" * 80)
    print("2連単 EV 分析結果")
    print("=" * 80)

    for (jyo_cd, jyo_name), jyo_df in pred_df.groupby(["jyo_cd", "jyo_name"]):
        print(f"\n{'━' * 80}")
        print(f"  {jyo_name} ({jyo_cd})")
        print(f"{'━' * 80}")
        print(f"  {'R':>3}  {'組合せ':^8}  {'モデル確率':>9}  "
              f"{'実際倍率':>8}  {'EV':>6}  {'判定':>8}")
        print(f"  {'─' * 60}")

        for _, row in jyo_df.sort_values("race_no").iterrows():
            odds_str = f"{row['actual_odds']:.1f}倍" if pd.notna(row["actual_odds"]) else "──"
            ev_str   = f"{row['ev']:.3f}" if pd.notna(row["ev"]) else "──"
            bet_str  = row.get("bet", "")
            print(
                f"  {int(row['race_no']):2d}R  "
                f"【{row['pred_1st']}-{row['pred_2nd']}】    "
                f"{row['prob_exacta']:>8.1%}  "
                f"{odds_str:>8}  {ev_str:>6}  {bet_str}"
            )

    # ── プラス EV レース ─────────────────────────────────────────────────────
    pos_ev = pred_df[pred_df["ev"].notna() & (pred_df["ev"] >= 1.0)].copy()
    print(f"\n{'=' * 80}")
    if len(pos_ev) == 0:
        print("  プラス期待値 (EV≥1.0) の組み合わせは見つかりませんでした。")
    else:
        print(f"  ◎ プラス期待値 (EV≥1.0) の買い目: {len(pos_ev)}件")
        print(f"{'=' * 80}")
        print(f"  {'場':>6} {'R':>3}  {'組合せ':^8}  "
              f"{'確率':>7}  {'倍率':>7}  {'EV':>6}")
        for _, row in pos_ev.sort_values("ev", ascending=False).iterrows():
            print(
                f"  {row['jyo_name']:>6} {int(row['race_no']):>3}R  "
                f"【{row['pred_1st']}-{row['pred_2nd']}】    "
                f"{row['prob_exacta']:>6.1%}  "
                f"{row['actual_odds']:>5.1f}倍  {row['ev']:>6.3f}"
            )
    print()
    print("  ※ EV=1.0: 損益均衡  EV>1.0: 統計的プラス収支期待  EV<1.0: 見送り推奨")


# ── メイン ────────────────────────────────────────────────────────────────────

def _parse_race_filter(races_str: str | None) -> list[int] | None:
    if races_str is None:
        return None
    if "-" in races_str:
        lo, hi = races_str.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in races_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="2連単オッズ取得 & EV 計算")
    parser.add_argument(
        "--pred", default=None,
        help="scrape_today.py が出力した予測CSV (省略時はオッズ表のみ表示)"
    )
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y%m%d"),
        help="対象日 YYYYMMDD (default: 今日)"
    )
    parser.add_argument(
        "--jcd", default=None,
        help="場コード (例: 06=浜名湖)。--pred 指定時は CSV から自動取得"
    )
    parser.add_argument(
        "--races", default=None,
        help="レース番号 (例: 1-6 または 1,3,5)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="全30組み合わせのオッズ表を表示する"
    )
    parser.add_argument(
        "--out", default=None,
        help="EV 結果CSVの保存先"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"2連単 オッズ & EV 分析  {args.date}")
    print("=" * 60)

    # ── モード1: 予測CSV + オッズ → EV 計算 ─────────────────────────────────
    if args.pred:
        if not os.path.exists(args.pred):
            print(f"エラー: 予測CSVが見つかりません: {args.pred}")
            sys.exit(1)

        pred_df = pd.read_csv(args.pred)
        print(f"予測データ読み込み: {len(pred_df)} レース (from {args.pred})")

        # 場コードと日付を CSV から取得
        jyo_cds = pred_df["jyo_cd"].astype(str).str.zfill(2).unique()
        date_str = args.date

        # CSVのファイル名から日付を推測 (例: predictions_20260322.csv)
        m = re.search(r"(\d{8})", os.path.basename(args.pred))
        if m:
            date_str = m.group(1)
            print(f"日付: {date_str} (CSVファイル名から取得)")

        pred_df["jyo_cd"] = pred_df["jyo_cd"].astype(str).str.zfill(2)

        all_pred_dfs = []
        for jyo_cd in jyo_cds:
            jyo_name = VENUE_NAMES.get(jyo_cd, jyo_cd)
            jyo_pred = pred_df[pred_df["jyo_cd"] == jyo_cd].copy()

            race_nos = sorted(jyo_pred["race_no"].astype(int).unique())
            if args.races:
                race_filter = _parse_race_filter(args.races)
                race_nos = [r for r in race_nos if r in race_filter]

            print(f"\n{jyo_name}({jyo_cd}) {len(race_nos)}レース分オッズ取得中...")
            odds_by_race = fetch_odds_for_races(jyo_cd, race_nos, date_str)

            jyo_pred = ev_for_predictions(jyo_pred, odds_by_race)
            all_pred_dfs.append(jyo_pred)

            # --all オプション: 各レースのオッズ表も表示
            if args.all:
                for rno in race_nos:
                    if rno in odds_by_race:
                        print_odds_table(rno, odds_by_race[rno], jyo_name)

        result_df = pd.concat(all_pred_dfs, ignore_index=True)
        print_ev_summary(result_df)

        # 保存
        out_path = args.out or os.path.join(
            CKPT_DIR, f"ev_result_{date_str}.csv"
        )
        result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nEV結果CSV: {out_path}")

    # ── モード2: 場・レース直接指定 → オッズ表示 ─────────────────────────────
    elif args.jcd:
        jyo_cd   = args.jcd.zfill(2)
        jyo_name = VENUE_NAMES.get(jyo_cd, jyo_cd)
        date_str = args.date
        race_nos = _parse_race_filter(args.races) or list(range(1, 13))

        print(f"\n{jyo_name}({jyo_cd}) {len(race_nos)}レース オッズ取得中...")
        odds_by_race = fetch_odds_for_races(jyo_cd, race_nos, date_str)

        for rno, odds_dict in sorted(odds_by_race.items()):
            if odds_dict:
                print_odds_table(rno, odds_dict, jyo_name)
                if not args.all:
                    # 最低オッズ (人気) と最高オッズ (穴) を表示
                    valid = {k: v for k, v in odds_dict.items() if v > 0}
                    if valid:
                        top_pop = min(valid, key=valid.get)
                        top_hole = max(valid, key=valid.get)
                        print(
                            f"  人気: {top_pop[0]}-{top_pop[1]} "
                            f"({valid[top_pop]:.1f}倍)  "
                            f"穴: {top_hole[0]}-{top_hole[1]} "
                            f"({valid[top_hole]:.1f}倍)"
                        )

        out_path = args.out or os.path.join(
            CKPT_DIR, f"odds_{jyo_cd}_{date_str}.csv"
        )
        # 全オッズを CSV に保存
        rows = []
        for rno, odds_dict in odds_by_race.items():
            for (c1, c2), mul in odds_dict.items():
                rows.append({
                    "jyo_cd": jyo_cd, "jyo_name": jyo_name,
                    "race_no": rno, "c1": c1, "c2": c2, "odds": mul
                })
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nオッズCSV: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
