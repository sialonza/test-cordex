#!/usr/bin/env python3
"""
run_today.py
当日の予測〜EV計算を一括実行するラッパー。

ワークフロー:
  Step 1: scrape_today.py  出走表スクレイプ + モデル予測 → predictions_{date}.csv
  Step 2: fetch_odds.py    リアルタイムオッズ取得 + EV計算 → ev_result_{date}.csv

使い方:
  python run_today.py                         # 当日全場 (スクレイプ → EV)
  python run_today.py --jcd 06               # 浜名湖のみ
  python run_today.py --jcd 06 --races 5-8  # 浜名湖 5〜8R
  python run_today.py --date 20260322        # 指定日
  python run_today.py --ev-only              # 既存の予測CSVでEVのみ再計算
  python run_today.py --skip-ev             # スクレイプ・予測のみ (EV取得なし)
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "real55k")


def run(cmd: list[str], label: str) -> int:
    """サブプロセスを実行して終了コードを返す。"""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="ボートレース当日予測 & EV 計算 ワンショット実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y%m%d"),
        help="対象日 YYYYMMDD (default: 今日)",
    )
    parser.add_argument(
        "--jcd", default=None,
        help="場コード (例: 06=浜名湖)。省略時は全場",
    )
    parser.add_argument(
        "--races", default=None,
        help="レース番号 (例: 1-6 または 1,3,5)。省略時は全レース",
    )
    parser.add_argument(
        "--ev-only", action="store_true",
        help="スクレイプをスキップして既存の予測CSVでEVのみ再計算",
    )
    parser.add_argument(
        "--skip-ev", action="store_true",
        help="EVオッズ取得をスキップして予測のみ実行",
    )
    parser.add_argument(
        "--pred", default=None,
        help="--ev-only 時に使う予測CSVパス (省略時は自動検索)",
    )
    args = parser.parse_args()

    date_str = args.date
    pred_csv = args.pred or os.path.join(CKPT_DIR, f"predictions_{date_str}.csv")

    python = sys.executable

    print("=" * 60)
    print(f"  ボートレース 当日パイプライン  {date_str}")
    print("=" * 60)

    # ── Step 1: スクレイプ + 予測 ──────────────────────────────────────────
    if not args.ev_only:
        cmd_scrape = [python, os.path.join(BASE_DIR, "scrape_today.py"),
                      "--date", date_str,
                      "--out",  pred_csv]
        if args.jcd:
            cmd_scrape += ["--jcd", args.jcd]
        if args.races:
            cmd_scrape += ["--races", args.races]

        rc = run(cmd_scrape, "Step 1/2  出走表スクレイプ + モデル予測")
        if rc != 0:
            print(f"\n[ERROR] scrape_today.py が異常終了しました (exit={rc})")
            sys.exit(rc)
    else:
        if not os.path.exists(pred_csv):
            print(f"\n[ERROR] 予測CSVが見つかりません: {pred_csv}")
            print("  --pred <path> で指定するか、先にスクレイプを実行してください。")
            sys.exit(1)
        print(f"\n[--ev-only] スクレイプをスキップ。予測CSV: {pred_csv}")

    # ── Step 2: オッズ取得 + EV 計算 ──────────────────────────────────────
    if not args.skip_ev:
        ev_csv = os.path.join(CKPT_DIR, f"ev_result_{date_str}.csv")
        cmd_ev = [python, os.path.join(BASE_DIR, "fetch_odds.py"),
                  "--pred", pred_csv,
                  "--date", date_str,
                  "--out",  ev_csv]
        if args.races:
            cmd_ev += ["--races", args.races]

        rc = run(cmd_ev, "Step 2/2  リアルタイムオッズ取得 & EV 計算")
        if rc != 0:
            print(f"\n[ERROR] fetch_odds.py が異常終了しました (exit={rc})")
            sys.exit(rc)
    else:
        print("\n[--skip-ev] EV計算をスキップしました。")

    # ── 完了サマリー ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  完了!")
    print("=" * 60)
    print(f"  予測CSV : {pred_csv}")
    if not args.skip_ev:
        ev_csv = os.path.join(CKPT_DIR, f"ev_result_{date_str}.csv")
        print(f"  EV CSV  : {ev_csv}")
    print()
    print("  次のステップ:")
    print("    backtest_ev.py  ← バックテストでROI確認")
    print("    fetch_odds.py   ← オッズ更新時に再実行 (--ev-only)")
    print()


if __name__ == "__main__":
    main()
