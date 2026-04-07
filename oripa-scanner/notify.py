"""
通知モジュール

+EV オリパ検出時に LINE Notify / Slack / コンソール / ログに通知。
環境変数:
  LINE_NOTIFY_TOKEN - LINE Notify トークン
  SLACK_WEBHOOK_URL - Slack Incoming Webhook URL
"""
import os
import json
import urllib.request
import urllib.parse
from datetime import datetime


def notify_console(message: str):
    print(f"\n{'='*60}\n[ALERT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{message}\n{'='*60}\n")


def notify_line(message: str):
    token = os.environ.get("LINE_NOTIFY_TOKEN")
    if not token:
        return False
    try:
        data = urllib.parse.urlencode({"message": message}).encode("utf-8")
        req = urllib.request.Request(
            "https://notify-api.line.me/api/notify",
            data=data,
            headers={"Authorization": f"Bearer {token}"},
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"  [WARN] LINE notify failed: {e}")
        return False


def notify_slack(message: str):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False
    try:
        data = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"  [WARN] Slack notify failed: {e}")
        return False


def notify_log(message: str):
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"alerts_{datetime.now().strftime('%Y-%m')}.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}]\n{message}\n\n")


def notify_all(message: str):
    """全チャネルに通知"""
    notify_console(message)
    notify_log(message)
    notify_line(message)
    notify_slack(message)
