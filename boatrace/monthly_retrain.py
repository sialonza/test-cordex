#!/usr/bin/env python3
"""
monthly_retrain.py
モデルを月次で再学習するオーケストレータ。

■ 実行フロー
  1. 前月の実績データをダウンロード (download_real.py)
  2. race_results_big.csv / race_payouts.csv に追記・重複除去
  3. download_extra.py — レーサー/モーター/場別統計を再計算
  4. convert_boatracecsv.py — 特徴量エンジニアリング + 分割
  5. チェックポイントをバックアップ
  6. run_train55k.py — LightGBM 学習
  7. calibrate.py — 確率キャリブレーション
  8. backtest_ev.py — テストセット EV バックテスト
  9. メトリクス比較・ログ保存

■ cron 設定 (毎月1日 03:00 に実行)
  0 3 1 * * cd /home/user/test-cordex && python boatrace/monthly_retrain.py >> boatrace/logs/retrain.log 2>&1

■ 手動実行
  python boatrace/monthly_retrain.py                   # 前月のデータで再学習
  python boatrace/monthly_retrain.py --month 2026-02   # 特定月を指定
  python boatrace/monthly_retrain.py --skip-download   # ダウンロードをスキップ (デバッグ用)
  python boatrace/monthly_retrain.py --dry-run         # ステップ確認のみ (実行なし)
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import logging
import textwrap
from datetime import datetime, timedelta
from calendar import monthrange
from pathlib import Path

import notify as _notify

# ── パス ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
LOG_DIR    = SCRIPT_DIR / "logs"
CKPT_DIR   = SCRIPT_DIR / "checkpoints" / "real55k"
RAW_DIR    = SCRIPT_DIR / "data" / "raw"
EXTRA_DIR  = SCRIPT_DIR / "data" / "extra"
HISTORY_FILE = SCRIPT_DIR / "retrain_history.json"

LOG_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# メトリクス変化の警告閾値 (前回比でこれ以上悪化したらアラート)
ALERT_AUC_DROP   = 0.02   # AUC が 2% 以上低下
ALERT_F1_DROP    = 0.03   # F1 が 3% 以上低下
ALERT_EXACTA_DROP = 0.01  # 2連単的中率が 1% 以上低下


# ── ロギング ──────────────────────────────────────────────────────────────────

def setup_logging(run_id: str) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("retrain")


# ── ユーティリティ ────────────────────────────────────────────────────────────

def run(cmd: list[str], dry_run: bool = False, logger: logging.Logger = None) -> bool:
    """サブプロセスを実行し、成否を返す。"""
    display = " ".join(cmd)
    if logger:
        logger.info(f"  $ {display}")
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        if logger:
            logger.error(f"  コマンド失敗 (exit={result.returncode}): {display}")
        return False
    return True


def prev_month(ref: datetime | None = None) -> tuple[str, str]:
    """前月の開始日・終了日を (YYYY-MM-DD, YYYY-MM-DD) で返す。"""
    ref = ref or datetime.now()
    first_of_current = ref.replace(day=1)
    last_of_prev = first_of_current - timedelta(days=1)
    first_of_prev = last_of_prev.replace(day=1)
    return first_of_prev.strftime("%Y-%m-%d"), last_of_prev.strftime("%Y-%m-%d")


# ── Step 1: 新規データのダウンロードと追記 ───────────────────────────────────

import pandas as pd  # noqa: E402  (subprocess 前に必要)

PAYOUT_COL_MAP = {
    # download_real.py の出力列 → race_payouts.csv の列
    "exacta_1st":    "rank1_course",
    "exacta_2nd":    "rank2_course",
    "payout_exacta": "payout_2nd",
}


def append_new_data(start: str, end: str,
                    dry_run: bool, logger: logging.Logger) -> bool:
    """
    download_real.py で取得した新規データを race_results_big.csv /
    race_payouts.csv に追記する。
    """
    if dry_run:
        logger.info(f"  [DRY-RUN] download_real.py --start {start} --end {end}")
        return True

    logger.info(f"  実績データ取得: {start} ~ {end}")
    ok = run(
        [sys.executable, str(SCRIPT_DIR / "download_real.py"),
         "--start", start, "--end", end],
        dry_run=dry_run, logger=logger
    )
    if not ok:
        return False

    # ── race_results_big.csv への追記 ────────────────────────────────────────
    real_path = RAW_DIR / "race_results_real.csv"
    big_path  = RAW_DIR / "race_results_big.csv"

    if not real_path.exists():
        logger.warning("  race_results_real.csv が見つかりません。スキップ。")
        return True

    new_df = pd.read_csv(real_path, dtype={"jyo_cd": str})
    logger.info(f"  新規レース結果: {len(new_df):,} 行 ({start} ~ {end})")

    if big_path.exists():
        old_df = pd.read_csv(big_path, dtype={"jyo_cd": str})
        dedup_keys = ["date", "jyo_cd", "race_no", "course"]
        combined = pd.concat([old_df, new_df], ignore_index=True)
        # 最新データ優先で重複除去
        combined = (combined
                    .sort_values("date")
                    .drop_duplicates(subset=dedup_keys, keep="last")
                    .reset_index(drop=True))
        n_added = len(combined) - len(old_df)
        logger.info(f"  race_results_big.csv: {len(old_df):,} → {len(combined):,} 行 (+{n_added:,})")
        combined.to_csv(big_path, index=False, encoding="utf-8-sig")
    else:
        new_df.to_csv(big_path, index=False, encoding="utf-8-sig")
        logger.info(f"  race_results_big.csv 新規作成: {len(new_df):,} 行")

    # ── race_payouts.csv への追記 ─────────────────────────────────────────────
    real_pay_path = RAW_DIR / "race_payouts_real.csv"
    pay_path      = RAW_DIR / "race_payouts.csv"

    if real_pay_path.exists():
        new_pay = pd.read_csv(real_pay_path, dtype={"jyo_cd": str})
        # 列名を正規化
        new_pay = new_pay.rename(columns=PAYOUT_COL_MAP)
        keep = ["date", "jyo_cd", "race_no", "rank1_course", "rank2_course", "payout_2nd"]
        new_pay = new_pay[[c for c in keep if c in new_pay.columns]]

        if pay_path.exists():
            old_pay = pd.read_csv(pay_path, dtype={"jyo_cd": str})
            combined_pay = (pd.concat([old_pay, new_pay], ignore_index=True)
                            .drop_duplicates(subset=["date", "jyo_cd", "race_no"], keep="last")
                            .sort_values("date")
                            .reset_index(drop=True))
            n_pay = len(combined_pay) - len(old_pay)
            logger.info(f"  race_payouts.csv: {len(old_pay):,} → {len(combined_pay):,} 行 (+{n_pay:,})")
            combined_pay.to_csv(pay_path, index=False, encoding="utf-8-sig")
        else:
            new_pay.to_csv(pay_path, index=False, encoding="utf-8-sig")
            logger.info(f"  race_payouts.csv 新規作成: {len(new_pay):,} 行")

    return True


# ── Step 5: チェックポイントのバックアップ ───────────────────────────────────

def backup_checkpoint(run_id: str, logger: logging.Logger) -> Path | None:
    """既存チェックポイントをタイムスタンプ付きフォルダにコピーする。"""
    if not CKPT_DIR.exists():
        return None
    backup_dir = SCRIPT_DIR / "checkpoints" / f"backup_{run_id}"
    shutil.copytree(CKPT_DIR, backup_dir)
    logger.info(f"  バックアップ: {backup_dir}")
    return backup_dir


def restore_checkpoint(backup_dir: Path, logger: logging.Logger) -> None:
    """バックアップから復元する (学習失敗時)。"""
    if backup_dir and backup_dir.exists():
        if CKPT_DIR.exists():
            shutil.rmtree(CKPT_DIR)
        shutil.copytree(backup_dir, CKPT_DIR)
        logger.warning(f"  チェックポイントを復元しました: {backup_dir}")


# ── メトリクス読み込み・比較 ─────────────────────────────────────────────────

def load_metrics() -> dict | None:
    """最新のメトリクスを checkpoints から読み込む。"""
    metrics = {}
    for name in ("lgbm_win", "lgbm_2nd", "lgbm_top3"):
        path = CKPT_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[name] = json.load(f)
    return metrics if metrics else None


def compare_metrics(prev: dict | None, curr: dict | None,
                    logger: logging.Logger) -> list[str]:
    """前回と今回のメトリクスを比較し、警告リストを返す。"""
    alerts = []
    if prev is None or curr is None:
        return alerts

    for model_name in ("lgbm_win", "lgbm_2nd"):
        if model_name not in prev or model_name not in curr:
            continue
        p = prev[model_name]
        c = curr[model_name]

        auc_prev = p.get("test_auc", 0)
        auc_curr = c.get("test_auc", 0)
        if auc_prev and auc_curr and (auc_prev - auc_curr) > ALERT_AUC_DROP:
            msg = (f"⚠ {model_name}: Test AUC が低下 "
                   f"{auc_prev:.4f} → {auc_curr:.4f} "
                   f"(-{auc_prev - auc_curr:.4f})")
            alerts.append(msg)
            logger.warning(msg)

        f1_prev = p.get("test_f1", 0)
        f1_curr = c.get("test_f1", 0)
        if f1_prev and f1_curr and (f1_prev - f1_curr) > ALERT_F1_DROP:
            msg = (f"⚠ {model_name}: Test F1 が低下 "
                   f"{f1_prev:.4f} → {f1_curr:.4f} "
                   f"(-{f1_prev - f1_curr:.4f})")
            alerts.append(msg)
            logger.warning(msg)

    # 2連単的中率
    for m_name in ("lgbm_win",):
        exacta_prev = (prev.get(m_name) or {}).get("test_exacta_hit_rate")
        exacta_curr = (curr.get(m_name) or {}).get("test_exacta_hit_rate")
        if exacta_prev and exacta_curr and (exacta_prev - exacta_curr) > ALERT_EXACTA_DROP:
            msg = (f"⚠ 2連単的中率が低下: "
                   f"{exacta_prev:.4f} → {exacta_curr:.4f} "
                   f"(-{exacta_prev - exacta_curr:.4f})")
            alerts.append(msg)
            logger.warning(msg)

    return alerts


def print_metrics_table(prev: dict | None, curr: dict | None,
                         logger: logging.Logger) -> None:
    """前回・今回メトリクスを並べて表示する。"""
    logger.info("\n  ─────── メトリクス比較 ───────")
    logger.info(f"  {'モデル':<12} {'指標':<12} {'前回':>8} {'今回':>8} {'差':>8}")
    logger.info(f"  {'─' * 54}")

    for model_name in ("lgbm_win", "lgbm_2nd", "lgbm_top3"):
        p = (prev or {}).get(model_name, {})
        c = (curr or {}).get(model_name, {})
        for metric in ("test_auc", "test_f1"):
            pv = p.get(metric)
            cv = c.get(metric)
            if pv is None and cv is None:
                continue
            diff_str = f"{cv - pv:+.4f}" if pv and cv else "  ─"
            logger.info(
                f"  {model_name:<12} {metric:<12} "
                f"{pv or 0:>8.4f} {cv or 0:>8.4f} {diff_str:>8}"
            )


# ── 履歴ファイル ─────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="月次再学習オーケストレータ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            例:
              python monthly_retrain.py                   前月データで再学習
              python monthly_retrain.py --month 2026-02   2月データで再学習
              python monthly_retrain.py --skip-download   ダウンロードをスキップ
              python monthly_retrain.py --dry-run         ステップ確認のみ

            cron 設定 (毎月1日 03:00):
              0 3 1 * * cd /home/user/test-cordex && \\
                python boatrace/monthly_retrain.py >> boatrace/logs/retrain.log 2>&1
        """)
    )
    parser.add_argument("--month", default=None,
                        help="対象月 YYYY-MM (default: 前月)")
    parser.add_argument("--skip-download", action="store_true",
                        help="データダウンロードをスキップ")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="バックテストをスキップ (高速化)")
    parser.add_argument("--dry-run", action="store_true",
                        help="実際には実行せず手順を表示")
    parser.add_argument("--no-backup", action="store_true",
                        help="チェックポイントのバックアップをスキップ")
    parser.add_argument("--no-notify", action="store_true",
                        help="外部通知 (Slack/LINE/Discord) をスキップ")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_id)

    # ── 対象期間 ─────────────────────────────────────────────────────────────
    if args.month:
        ref = datetime.strptime(args.month + "-01", "%Y-%m-%d")
        last_day = monthrange(ref.year, ref.month)[1]
        start = ref.strftime("%Y-%m-%d")
        end   = ref.replace(day=last_day).strftime("%Y-%m-%d")
    else:
        start, end = prev_month()

    logger.info("=" * 64)
    logger.info(f"月次再学習  run_id={run_id}")
    logger.info(f"対象期間: {start} ~ {end}")
    logger.info("=" * 64)

    if args.dry_run:
        logger.info("[DRY-RUN モード: 実際には実行しません]")

    # ── 前回メトリクスを退避 ─────────────────────────────────────────────────
    prev_metrics = load_metrics()

    backup_dir = None

    try:
        # ─ Step 1: データダウンロード & 追記 ──────────────────────────────
        if not args.skip_download:
            logger.info("\n[Step 1] データダウンロード & 追記")
            ok = append_new_data(start, end, args.dry_run, logger)
            if not ok:
                logger.error("  データ取得に失敗しました。再学習を中止します。")
                sys.exit(1)
        else:
            logger.info("\n[Step 1] データダウンロード → スキップ")

        # ─ Step 2: 追加統計の再計算 ───────────────────────────────────────
        logger.info("\n[Step 2] download_extra.py — 統計再計算")
        if not run([sys.executable, str(SCRIPT_DIR / "download_extra.py")],
                   dry_run=args.dry_run, logger=logger):
            logger.error("  download_extra.py 失敗。再学習を中止します。")
            sys.exit(1)

        # ─ Step 2.5: 市場払戻統計の更新 ───────────────────────────────────
        logger.info("\n[Step 2.5] generate_payout_stats.py — 払戻統計更新")
        if not run([sys.executable, str(SCRIPT_DIR / "generate_payout_stats.py")],
                   dry_run=args.dry_run, logger=logger):
            logger.warning("  generate_payout_stats.py 失敗。前回の統計で続行します。")

        # ─ Step 3: 特徴量エンジニアリング ────────────────────────────────
        logger.info("\n[Step 3] convert_boatracecsv.py — 特徴量生成")
        if not run([sys.executable, str(SCRIPT_DIR / "convert_boatracecsv.py")],
                   dry_run=args.dry_run, logger=logger):
            logger.error("  convert_boatracecsv.py 失敗。再学習を中止します。")
            sys.exit(1)

        # ─ Step 4: チェックポイントのバックアップ ────────────────────────
        if not args.no_backup and not args.dry_run:
            logger.info("\n[Step 4] チェックポイントをバックアップ")
            backup_dir = backup_checkpoint(run_id, logger)
        else:
            logger.info("\n[Step 4] バックアップ → スキップ")

        # ─ Step 5: モデル学習 ────────────────────────────────────────────
        logger.info("\n[Step 5] run_train55k.py — LightGBM 学習")
        if not run([sys.executable, str(SCRIPT_DIR / "run_train55k.py")],
                   dry_run=args.dry_run, logger=logger):
            logger.error("  run_train55k.py 失敗。バックアップから復元します。")
            restore_checkpoint(backup_dir, logger)
            sys.exit(1)

        # ─ Step 6: 確率キャリブレーション ───────────────────────────────
        logger.info("\n[Step 6] calibrate.py — 確率キャリブレーション")
        if not run([sys.executable, str(SCRIPT_DIR / "calibrate.py")],
                   dry_run=args.dry_run, logger=logger):
            logger.warning("  calibrate.py 失敗。キャリブレーションなしで続行します。")

        # ─ Step 7: バックテスト ──────────────────────────────────────────
        if not args.skip_backtest:
            logger.info("\n[Step 7] backtest_ev.py — EV バックテスト")
            run([sys.executable, str(SCRIPT_DIR / "backtest_ev.py")],
                dry_run=args.dry_run, logger=logger)
        else:
            logger.info("\n[Step 7] バックテスト → スキップ")

        # ─ Step 8: メトリクス比較 & ログ ─────────────────────────────────
        logger.info("\n[Step 8] メトリクス比較")
        curr_metrics = load_metrics() if not args.dry_run else None
        print_metrics_table(prev_metrics, curr_metrics, logger)
        alerts = compare_metrics(prev_metrics, curr_metrics, logger)

        # ─ 履歴保存 ──────────────────────────────────────────────────────
        history = load_history()
        record = {
            "run_id":    run_id,
            "start":     start,
            "end":       end,
            "timestamp": datetime.now().isoformat(),
            "metrics":   curr_metrics or {},
            "alerts":    alerts,
            "dry_run":   args.dry_run,
        }
        history.append(record)
        if not args.dry_run:
            save_history(history)

        # ─ サマリー ──────────────────────────────────────────────────────
        logger.info("\n" + "=" * 64)
        if alerts:
            logger.warning(f"再学習完了 ⚠ {len(alerts)} 件の警告あり")
            for a in alerts:
                logger.warning(f"  {a}")
        else:
            logger.info("再学習完了 ✓ メトリクスは前回と同等以上")
        logger.info(f"チェックポイント: {CKPT_DIR}")
        if backup_dir:
            logger.info(f"バックアップ:     {backup_dir}")
        logger.info("=" * 64)

        # ─ 外部通知 ──────────────────────────────────────────────────────
        if not args.no_notify and not args.dry_run:
            status = "warning" if alerts else "success"
            logger.info("\n[通知] 外部チャンネルに送信中...")
            _notify.send(
                status=status,
                run_id=run_id,
                period=f"{start} ~ {end}",
                metrics=curr_metrics,
                alerts=alerts,
            )

    except KeyboardInterrupt:
        logger.warning("\n中断されました。バックアップから復元します。")
        restore_checkpoint(backup_dir, logger)
        if not args.no_notify:
            _notify.send(
                status="error",
                run_id=run_id,
                period=f"{start} ~ {end}",
                error_msg="KeyboardInterrupt — 手動中断",
            )
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n予期しないエラー: {e}")
        restore_checkpoint(backup_dir, logger)
        if not args.no_notify:
            _notify.send(
                status="error",
                run_id=run_id,
                period=f"{start} ~ {end}",
                error_msg=str(e),
            )
        raise


if __name__ == "__main__":
    main()
