#!/usr/bin/env python3
"""
notify.py
再学習アラート / 完了通知を外部チャンネルに送信する。

対応チャンネル:
  - Slack        (SLACK_WEBHOOK_URL)
  - LINE Notify  (LINE_NOTIFY_TOKEN)
  - Discord      (DISCORD_WEBHOOK_URL)

設定方法:
  1. 環境変数でトークンを設定する (推奨)
       export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
       export LINE_NOTIFY_TOKEN=xxxxx
       export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx

  2. または boatrace/notify_config.json に記述する
       {
         "slack_webhook_url": "https://hooks.slack.com/services/xxx",
         "line_notify_token": "xxxxx",
         "discord_webhook_url": "https://discord.com/api/webhooks/xxx"
       }
       ※ notify_config.json は .gitignore に追加し、トークンを公開しないこと。

単体テスト:
  python notify.py --test          テスト通知を全チャンネルに送信
  python notify.py --test --slack  Slack のみ
"""

import os
import sys
import json
import logging
import argparse
import textwrap
import requests
from pathlib import Path
from datetime import datetime

BASE_DIR    = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "notify_config.json"
LOG         = logging.getLogger("notify")


# ── 設定読み込み ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """環境変数 → notify_config.json の順で設定を読み込む。"""
    cfg = {}

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
        except Exception:
            pass

    # 環境変数が優先
    if os.getenv("SLACK_WEBHOOK_URL"):
        cfg["slack_webhook_url"] = os.getenv("SLACK_WEBHOOK_URL")
    if os.getenv("LINE_NOTIFY_TOKEN"):
        cfg["line_notify_token"] = os.getenv("LINE_NOTIFY_TOKEN")
    if os.getenv("DISCORD_WEBHOOK_URL"):
        cfg["discord_webhook_url"] = os.getenv("DISCORD_WEBHOOK_URL")

    return cfg


# ── チャンネル別送信 ──────────────────────────────────────────────────────────

def _send_slack(webhook_url: str, text: str) -> bool:
    """Slack Incoming Webhook に送信する。"""
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        if r.status_code == 200:
            return True
        LOG.warning(f"Slack HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        LOG.warning(f"Slack 送信エラー: {e}")
    return False


def _send_line(token: str, message: str) -> bool:
    """LINE Notify に送信する。"""
    try:
        r = requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message},
            timeout=10,
        )
        if r.status_code == 200:
            return True
        LOG.warning(f"LINE HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        LOG.warning(f"LINE 送信エラー: {e}")
    return False


def _send_discord(webhook_url: str, text: str) -> bool:
    """Discord Webhook に送信する。"""
    try:
        r = requests.post(webhook_url, json={"content": text}, timeout=10)
        if r.status_code in (200, 204):
            return True
        LOG.warning(f"Discord HTTP {r.status_code}: {r.text[:100]}")
    except Exception as e:
        LOG.warning(f"Discord 送信エラー: {e}")
    return False


# ── メッセージ生成 ────────────────────────────────────────────────────────────

def _build_message(
    status: str,            # "success" | "warning" | "error"
    run_id: str,
    period: str,            # "2026-02-01 ~ 2026-02-28"
    metrics: dict | None,
    alerts: list[str],
    error_msg: str | None = None,
) -> str:
    icon = {"success": "✅", "warning": "⚠️", "error": "❌"}.get(status, "ℹ️")
    label = {"success": "完了", "warning": "完了 (警告あり)", "error": "失敗"}.get(status, status)

    lines = [
        f"{icon} [BoatRace] 月次再学習 {label}",
        f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"期間: {period}",
        f"run_id: {run_id}",
        "─" * 36,
    ]

    # メトリクス
    if metrics:
        lines.append("メトリクス:")
        for model_name in ("lgbm_win", "lgbm_2nd"):
            m = metrics.get(model_name, {})
            if not m:
                continue
            auc = m.get("test_auc")
            f1  = m.get("test_f1")
            auc_s = f"{auc:.4f}" if auc is not None else "─"
            f1_s  = f"{f1:.4f}"  if f1  is not None else "─"
            lines.append(f"  {model_name}: AUC={auc_s}  F1={f1_s}")
    else:
        lines.append("メトリクス: (取得なし)")

    lines.append("─" * 36)

    # アラート
    if alerts:
        lines.append(f"警告 ({len(alerts)}件):")
        for a in alerts:
            lines.append(f"  {a}")
    elif status != "error":
        lines.append("警告: なし")

    # エラー
    if error_msg:
        lines.append(f"エラー: {error_msg}")

    return "\n".join(lines)


# ── 公開 API ─────────────────────────────────────────────────────────────────

def send(
    status: str,
    run_id: str,
    period: str,
    metrics: dict | None = None,
    alerts: list[str] | None = None,
    error_msg: str | None = None,
    channels: list[str] | None = None,
) -> dict[str, bool]:
    """
    通知を送信する。

    Parameters
    ----------
    status   : "success" | "warning" | "error"
    run_id   : monthly_retrain の run_id
    period   : "YYYY-MM-DD ~ YYYY-MM-DD"
    metrics  : load_metrics() の戻り値
    alerts   : compare_metrics() の戻り値
    error_msg: エラー時のメッセージ
    channels : ["slack", "line", "discord"] のサブセット (None = 全チャンネル)

    Returns
    -------
    {"slack": True/False, "line": True/False, "discord": True/False}
    """
    alerts = alerts or []
    cfg    = _load_config()
    msg    = _build_message(status, run_id, period, metrics, alerts, error_msg)

    results: dict[str, bool] = {}

    def _should(ch: str) -> bool:
        return channels is None or ch in channels

    if _should("slack") and cfg.get("slack_webhook_url"):
        results["slack"] = _send_slack(cfg["slack_webhook_url"], msg)
        LOG.info(f"  Slack 通知: {'OK' if results['slack'] else 'NG'}")

    if _should("line") and cfg.get("line_notify_token"):
        results["line"] = _send_line(cfg["line_notify_token"], "\n" + msg)
        LOG.info(f"  LINE 通知:  {'OK' if results['line'] else 'NG'}")

    if _should("discord") and cfg.get("discord_webhook_url"):
        results["discord"] = _send_discord(cfg["discord_webhook_url"], msg)
        LOG.info(f"  Discord 通知: {'OK' if results['discord'] else 'NG'}")

    if not results:
        LOG.debug("  通知設定なし (SLACK_WEBHOOK_URL / LINE_NOTIFY_TOKEN / DISCORD_WEBHOOK_URL)")

    return results


# ── CLI テスト ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="再学習アラート通知テスト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            例:
              python notify.py --test              全チャンネルにテスト通知
              python notify.py --test --slack      Slack のみ
              python notify.py --test --warning    警告ありのテスト
              python notify.py --test --error      エラーのテスト
        """),
    )
    parser.add_argument("--test",    action="store_true", help="テスト通知を送信")
    parser.add_argument("--slack",   action="store_true", help="Slack のみ")
    parser.add_argument("--line",    action="store_true", help="LINE のみ")
    parser.add_argument("--discord", action="store_true", help="Discord のみ")
    parser.add_argument("--warning", action="store_true", help="警告ありのテスト")
    parser.add_argument("--error",   action="store_true", help="エラーのテスト")
    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    channels = None
    if args.slack or args.line or args.discord:
        channels = []
        if args.slack:   channels.append("slack")
        if args.line:    channels.append("line")
        if args.discord: channels.append("discord")

    dummy_metrics = {
        "lgbm_win": {"test_auc": 0.7234, "test_f1": 0.4521},
        "lgbm_2nd": {"test_auc": 0.6987, "test_f1": 0.4102},
    }

    if args.error:
        status, alerts, error_msg = "error", [], "run_train55k.py が exit code 1 で終了"
    elif args.warning:
        status = "warning"
        alerts = [
            "⚠ lgbm_win: Test AUC が低下 0.7234 → 0.7100 (-0.0134)",
            "⚠ lgbm_2nd: Test F1 が低下 0.4102 → 0.3800 (-0.0302)",
        ]
        error_msg = None
    else:
        status, alerts, error_msg = "success", [], None

    results = send(
        status=status,
        run_id="20260301_030000",
        period="2026-02-01 ~ 2026-02-28",
        metrics=dummy_metrics if not args.error else None,
        alerts=alerts,
        error_msg=error_msg,
        channels=channels,
    )

    cfg = _load_config()
    configured = [k.split("_")[0] for k in
                  ("slack_webhook_url", "line_notify_token", "discord_webhook_url")
                  if cfg.get(k)]

    print("\n設定済みチャンネル:", configured or ["なし"])
    print("送信結果:", results or {"(なし)": "─"})
    if not configured:
        print("\n※ 環境変数または notify_config.json でトークンを設定してください。")


if __name__ == "__main__":
    main()
