#!/usr/bin/env python3
"""
Big Move Predictor フォワードテスト
4H-C推奨設定でリアルタイム監視 + シグナルログ記録

Usage:
  python3 forward-test.py              # 現在のシグナルチェック
  python3 forward-test.py --status     # フォワードテスト成績表示
  python3 forward-test.py --reset      # ログリセット
"""
import sys
import os
import json
import urllib.request
from datetime import datetime, timezone

# backtest.py から共有関数をインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import (
    fetch_ohlcv, calc_rsi, calc_atr, calc_ema, calc_bb, calc_kc,
    calc_squeeze_score, calc_direction
)

# ============================================================
#  推奨設定: 4H-C 広幅 (バックテストS判定)
# ============================================================
CONFIG = {
    "name":         "4H-C 広幅",
    "symbol":       "BTC_USDT",
    "timeframe":    "4h",
    "min_score":    30,
    "min_dir_prob": 55,
    "sl_atr_mult":  2.5,
    "tp_atr_mult":  3.0,
    "trail_atr_mult": 2.0,
    "max_hold":     15,
    "slippage":     0.001,
    "commission":   0.001,
}

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward-log.json")


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {"config": CONFIG["name"], "started": None, "signals": [], "trades": []}


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def scan_current_signal():
    """最新データを取得し、現在シグナルが出ているかチェック"""
    cfg = CONFIG
    print(f"{'='*60}")
    print(f"  Big Move Predictor フォワードテスト")
    print(f"  設定: {cfg['name']}")
    print(f"  {cfg['symbol']} {cfg['timeframe']}")
    print(f"  Score≥{cfg['min_score']} Prob≥{cfg['min_dir_prob']}")
    print(f"  SL:{cfg['sl_atr_mult']}×ATR TP:{cfg['tp_atr_mult']}×ATR Trail:{cfg['trail_atr_mult']}×ATR")
    print(f"{'='*60}")

    print(f"\nデータ取得中...")
    data = fetch_ohlcv(cfg["symbol"], cfg["timeframe"])
    if not data or len(data) < 50:
        print("[ERROR] データ取得失敗")
        return

    closes = [d["c"] for d in data]
    highs = [d["h"] for d in data]
    lows = [d["l"] for d in data]
    opens = [d["o"] for d in data]
    vols = [d["v"] for d in data]
    n = len(data)

    # Indicators
    rsi = calc_rsi(closes)
    atr = calc_atr(highs, lows, closes)
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    bbU, bbB, bbL = calc_bb(closes)
    kcU, kcM, kcL = calc_kc(closes, highs, lows)
    volSMA = calc_ema(vols, 20)

    # 最新バーを分析
    i = n - 1
    ts = data[i].get("ts", 0)
    ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts > 1000 else str(ts)

    score, kcSqz, bbSqz, atrComp = calc_squeeze_score(
        i, closes, highs, lows, opens, vols,
        bbU, bbB, bbL, kcU, kcM, kcL, atr, rsi, ema9, ema21, volSMA)

    isBull, dirProb, bullW, bearW = calc_direction(
        i, closes, highs, lows, opens, vols,
        rsi, ema9, ema21, atr, bbU, bbL, volSMA, is_daily=False)

    print(f"\n--- 最新バー分析 ({ts_str}) ---")
    print(f"  価格:     ${closes[i]:,.2f}")
    print(f"  ATR(14):  ${atr[i]:,.2f}")
    print(f"  RSI(14):  {rsi[i]:.1f}")
    print(f"  EMA9:     ${ema9[i]:,.2f}")
    print(f"  EMA21:    ${ema21[i]:,.2f}")

    print(f"\n--- スクイーズ分析 ---")
    print(f"  スコア:   {score}%  (閾値: {cfg['min_score']}%)")
    print(f"  KC squeeze: {'YES' if kcSqz else 'NO'}")
    print(f"  BB squeeze: {'YES' if bbSqz else 'NO'}")
    print(f"  ATR圧縮:    {atrComp:.2f}")

    print(f"\n--- 方向予測 ---")
    direction = "LONG" if isBull else "SHORT"
    print(f"  方向:     {direction}")
    print(f"  確信度:   {dirProb}%  (閾値: {cfg['min_dir_prob']}%)")
    print(f"  Bull重み: {bullW:.1f}")
    print(f"  Bear重み: {bearW:.1f}")

    # トレンドフィルタチェック
    trend_ok = True
    if i >= 5:
        ema_slope = ema21[i] - ema21[i - 5]
        if isBull and ema_slope < -closes[i] * 0.001:
            trend_ok = False
            print(f"  トレンド: BLOCKED (EMA21下降中にLong)")
        elif not isBull and ema_slope > closes[i] * 0.001:
            trend_ok = False
            print(f"  トレンド: BLOCKED (EMA21上昇中にShort)")
        else:
            print(f"  トレンド: OK")

    # シグナル判定
    has_signal = (score >= cfg["min_score"] and
                  dirProb >= cfg["min_dir_prob"] and
                  trend_ok)

    print(f"\n{'='*60}")
    if has_signal:
        entry_atr = atr[i]
        swing_lookback = 10
        if isBull:
            entry_est = closes[i] * (1 + cfg["slippage"])
            atr_sl = entry_est - entry_atr * cfg["sl_atr_mult"]
            swing_sl = min(lows[max(0, i - swing_lookback):i + 1])
            sl = min(atr_sl, swing_sl)
            tp = entry_est + entry_atr * cfg["tp_atr_mult"]
        else:
            entry_est = closes[i] * (1 - cfg["slippage"])
            atr_sl = entry_est + entry_atr * cfg["sl_atr_mult"]
            swing_sl = max(highs[max(0, i - swing_lookback):i + 1])
            sl = max(atr_sl, swing_sl)
            tp = entry_est - entry_atr * cfg["tp_atr_mult"]

        risk_pct = abs(entry_est - sl) / entry_est * 100
        reward_pct = abs(tp - entry_est) / entry_est * 100

        print(f"  >>> SIGNAL: {direction} <<<")
        print(f"  スコア: {score}%  確信度: {dirProb}%")
        print(f"")
        print(f"  エントリー目安: ${entry_est:,.2f}")
        print(f"  SL:             ${sl:,.2f}  (-{risk_pct:.2f}%)")
        print(f"  TP:             ${tp:,.2f}  (+{reward_pct:.2f}%)")
        print(f"  RR比:           {reward_pct/risk_pct:.2f}")
        print(f"  最大保有:       {cfg['max_hold']} bars ({cfg['max_hold']*4}時間)")
        print(f"")
        print(f"  *** 次の4H足が{direction}方向を確認してからエントリー ***")

        # ログに記録
        log = load_log()
        if not log["started"]:
            log["started"] = datetime.now(timezone.utc).isoformat()

        signal = {
            "ts": ts_str,
            "price": closes[i],
            "score": score,
            "dir": direction,
            "dirProb": dirProb,
            "entry_est": round(entry_est, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "rr": round(reward_pct / risk_pct, 2),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        log["signals"].append(signal)
        save_log(log)
        print(f"  [LOG] シグナルを {LOG_FILE} に記録済み")
    else:
        print(f"  シグナルなし")
        reasons = []
        if score < cfg["min_score"]:
            reasons.append(f"スコア不足({score}<{cfg['min_score']})")
        if dirProb < cfg["min_dir_prob"]:
            reasons.append(f"確信度不足({dirProb}<{cfg['min_dir_prob']})")
        if not trend_ok:
            reasons.append("トレンドフィルタ")
        print(f"  理由: {', '.join(reasons)}")
        print(f"  → 次の4H足を待機")

    print(f"{'='*60}")

    # 直近5本のスコア履歴
    print(f"\n--- 直近5本のスコア ---")
    for j in range(max(0, n - 5), n):
        s, _, _, _ = calc_squeeze_score(
            j, closes, highs, lows, opens, vols,
            bbU, bbB, bbL, kcU, kcM, kcL, atr, rsi, ema9, ema21, volSMA)
        b, dp, _, _ = calc_direction(
            j, closes, highs, lows, opens, vols,
            rsi, ema9, ema21, atr, bbU, bbL, volSMA, is_daily=False)
        t = data[j].get("ts", 0)
        t_s = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%m-%d %H:%M") if t > 1000 else str(t)
        d = "L" if b else "S"
        flag = " <<<" if s >= cfg["min_score"] and dp >= cfg["min_dir_prob"] else ""
        print(f"  {t_s}  ${closes[j]:>10,.2f}  Score:{s:>3}  {d} {dp}%{flag}")


def show_status():
    """フォワードテストの成績を表示"""
    log = load_log()
    signals = log.get("signals", [])
    trades = log.get("trades", [])

    print(f"{'='*60}")
    print(f"  フォワードテスト成績")
    print(f"{'='*60}")
    print(f"  設定:     {log.get('config', 'N/A')}")
    print(f"  開始:     {log.get('started', '未開始')}")
    print(f"  シグナル: {len(signals)}件")
    print(f"  トレード: {len(trades)}件")

    if signals:
        print(f"\n--- シグナル履歴 ---")
        for s in signals[-10:]:  # 最新10件
            print(f"  {s['ts']}  {s['dir']} ${s['price']:,.2f}  "
                  f"Score:{s['score']} Prob:{s['dirProb']}%  RR:{s['rr']}")

    if trades:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = sum(pnls)
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

        print(f"\n--- トレード成績 ---")
        print(f"  勝率:         {wr:.1f}%")
        print(f"  PF:           {pf:.2f}")
        print(f"  累積リターン: {total*100:.2f}%")
        for t in trades[-10:]:
            result = "WIN" if t["pnl"] > 0 else "LOSS"
            print(f"  {t.get('ts','')}  {t['dir']} {result} {t['pnl']*100:+.2f}%  ({t.get('exit_reason','')})")
    elif signals:
        print(f"\n  トレードなし（シグナル記録のみ）")
        print(f"  手動で結果を追加: forward-log.json の trades 配列に記入")

    print(f"\n  ログファイル: {LOG_FILE}")


def record_trade():
    """直近シグナルの結果を手動記録"""
    log = load_log()
    signals = log.get("signals", [])
    if not signals:
        print("シグナルがありません")
        return

    last = signals[-1]
    print(f"直近シグナル: {last['ts']} {last['dir']} ${last['price']:,.2f}")
    print(f"  Entry: ${last['entry_est']:,.2f}  SL: ${last['sl']:,.2f}  TP: ${last['tp']:,.2f}")

    result = input("結果 (w=win, l=loss, s=skip): ").strip().lower()
    if result == "s":
        print("スキップ")
        return

    exit_price = input("決済価格 ($): ").strip()
    try:
        exit_price = float(exit_price.replace(",", ""))
    except ValueError:
        print("無効な価格")
        return

    exit_reason = input("理由 (TP/SL/TIME/MANUAL): ").strip().upper() or "MANUAL"

    if last["dir"] == "LONG":
        pnl = (exit_price - last["entry_est"]) / last["entry_est"] - CONFIG["commission"] * 2
    else:
        pnl = (last["entry_est"] - exit_price) / last["entry_est"] - CONFIG["commission"] * 2

    trade = {
        "ts": last["ts"],
        "dir": last["dir"],
        "entry": last["entry_est"],
        "exit": exit_price,
        "pnl": round(pnl, 6),
        "exit_reason": exit_reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    log.setdefault("trades", []).append(trade)
    save_log(log)
    print(f"記録完了: {trade['dir']} PnL {pnl*100:+.2f}% ({exit_reason})")


# ============================================================
#  メイン
# ============================================================
if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--reset" in sys.argv:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            print(f"ログをリセットしました: {LOG_FILE}")
        else:
            print("ログファイルなし")
    elif "--record" in sys.argv:
        record_trade()
    else:
        scan_current_signal()
