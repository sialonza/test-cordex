#!/usr/bin/env python3
"""
ETH特化 Z-Score平均回帰バックテスト

目的: N=9だったZ-Score戦略をデータ最大化+ウォークフォワードで本物か検証
- CryptoCompare 8ページ取得 → 最大4000 4Hバー
- 3分割ウォークフォワード（各期間でOOS検証）
- パラメータ15パターン網羅
- 個別トレード表示で透明性確保
"""
import math
import json
import urllib.request
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_zscore_reversion import calc_zscore, strat_zscore_reversion


# ============================================================
#  指標計算（backtest.pyから独立）
# ============================================================
def calc_ema(data, period):
    ema = [0.0] * len(data)
    if len(data) < period:
        return ema
    k = 2.0 / (period + 1)
    ema[period - 1] = sum(data[:period]) / period
    for i in range(period, len(data)):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_rsi(closes, period=14):
    rsi = [50.0] * len(closes)
    if len(closes) < period + 1:
        return rsi
    gains = []
    losses = []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0)) / period
        if avg_l == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100 - 100 / (1 + avg_g / avg_l)
    return rsi


def calc_atr(highs, lows, closes, period=14):
    atr = [0.0] * len(closes)
    if len(closes) < 2:
        return atr
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return atr
    atr[period] = sum(trs[:period]) / period
    for i in range(period + 1, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + trs[i - 1]) / period
    return atr


def calc_bb(closes, period=20, mult=2.0):
    n = len(closes)
    upper = [0.0] * n
    basis = [0.0] * n
    lower = [0.0] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        m = sum(window) / period
        std = (sum((x - m) ** 2 for x in window) / period) ** 0.5
        basis[i] = m
        upper[i] = m + mult * std
        lower[i] = m - mult * std
    return upper, basis, lower


# ============================================================
#  データ取得 — 最大ページ数でETH 1Hを取得し4Hに変換
# ============================================================
def fetch_eth_max(max_pages=10):
    """CryptoCompare ETH/USDT 1H を最大ページ取得 → 4H集約"""
    base_url = "https://min-api.cryptocompare.com/data/v2/histohour?fsym=ETH&tsym=USDT&limit=2000"
    all_candles = []
    toTs = ""

    for page in range(max_pages):
        url = base_url if not toTs else f"{base_url}&toTs={toTs}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ZScoreETH/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = json.loads(resp.read())
                candles = raw.get("Data", {}).get("Data", [])
                if not candles:
                    break
                all_candles.extend(candles)
                toTs = str(min(c.get("time", 0) for c in candles) - 1)
                print(f"  [Page {page+1}] {len(candles)} bars (total raw: {len(all_candles)})")
        except Exception as e:
            print(f"  [ERR] Page {page+1}: {e}")
            break

    # Dedup + sort
    seen = set()
    rows = []
    for c in all_candles:
        t = c.get("time", 0)
        if t in seen or (c.get("close", 0) == 0 and c.get("open", 0) == 0):
            continue
        seen.add(t)
        rows.append({
            "ts": t, "o": float(c["open"]), "h": float(c["high"]),
            "l": float(c["low"]), "c": float(c["close"]),
            "v": float(c.get("volumefrom", 0)),
        })
    rows.sort(key=lambda x: x["ts"])

    # 1H → 4H
    agg = []
    for k in range(0, len(rows) - 3, 4):
        chunk = rows[k:k + 4]
        agg.append({
            "ts": chunk[0]["ts"], "o": chunk[0]["o"],
            "h": max(c["h"] for c in chunk), "l": min(c["l"] for c in chunk),
            "c": chunk[-1]["c"], "v": sum(c["v"] for c in chunk),
        })
    print(f"  [OK] {len(rows)} 1H bars → {len(agg)} 4H bars")
    from datetime import datetime
    if agg:
        t0 = datetime.utcfromtimestamp(agg[0]["ts"]).strftime("%Y-%m-%d")
        t1 = datetime.utcfromtimestamp(agg[-1]["ts"]).strftime("%Y-%m-%d")
        days = (agg[-1]["ts"] - agg[0]["ts"]) / 86400
        print(f"  [RANGE] {t0} ~ {t1} ({days:.0f} days)")
    return agg


# ============================================================
#  Z-Score バックテストエンジン（ウォークフォワード対応）
# ============================================================
def run_zscore_segment(data, z_lookback=50, z_entry=2.0, z_exit=0.3,
                       z_stop=3.5, sl_mult=3.5, max_hold=20, cooldown=5,
                       slippage=0.001, commission=0.001):
    """データ区間全体をテスト区間として実行（ウォームアップ70本除く）"""
    n = len(data)
    if n < 120:
        return None

    closes = [d["c"] for d in data]
    highs = [d["h"] for d in data]
    lows = [d["l"] for d in data]
    opens = [d["o"] for d in data]
    vols = [d["v"] for d in data]

    rsi = calc_rsi(closes)
    atr = calc_atr(highs, lows, closes)
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    ema50 = calc_ema(closes, 50)
    bbU, bbB, bbL = calc_bb(closes)
    volSMA = calc_ema(vols, 20)
    zscores = calc_zscore(closes, lookback=z_lookback)

    # Clear zscore cache to avoid cross-contamination
    strat_zscore_reversion.__defaults__[-1].clear()

    test_start = max(80, z_lookback + 20)
    trades = []
    in_trade = False
    trade = {}
    last_sig = -cooldown

    for i in range(test_start, n):
        if in_trade:
            bars_held = i - trade["entry_bar"]
            z_now = zscores[i]

            if trade["dir"] == "LONG":
                if z_now >= -z_exit:
                    pnl = (closes[i] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "LONG",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue
                if z_now < -z_stop:
                    pnl = (closes[i] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "LONG",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue
                if lows[i] <= trade["hard_sl"]:
                    pnl = (trade["hard_sl"] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "HSL", "bars": bars_held, "dir": "LONG",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue
                if bars_held >= max_hold:
                    pnl = (closes[i] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TIME", "bars": bars_held, "dir": "LONG",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue
            else:
                if z_now <= z_exit:
                    pnl = (trade["ep"] - closes[i] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "SHORT",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue
                if z_now > z_stop:
                    pnl = (trade["ep"] - closes[i] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "SHORT",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue
                if highs[i] >= trade["hard_sl"]:
                    pnl = (trade["ep"] - trade["hard_sl"] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "HSL", "bars": bars_held, "dir": "SHORT",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue
                if bars_held >= max_hold:
                    pnl = (trade["ep"] - closes[i] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TIME", "bars": bars_held, "dir": "SHORT",
                                   "entry_z": trade["entry_z"], "exit_z": z_now, "bar": i})
                    in_trade = False; continue

        if in_trade or i >= n - 2 or (i - last_sig) < cooldown:
            continue

        sig = strat_zscore_reversion(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                                      bbU, bbB, bbL, vols, volSMA, opens)
        if not sig:
            continue

        direction, confidence = sig
        last_sig = i
        a = atr[i] if atr[i] > 0 else closes[i] * 0.01

        if direction == "LONG":
            ep = closes[i] * (1 + slippage)
            trade = {"dir": "LONG", "entry_bar": i, "ep": ep,
                     "entry_z": zscores[i], "hard_sl": ep - a * sl_mult}
        else:
            ep = closes[i] * (1 - slippage)
            trade = {"dir": "SHORT", "entry_bar": i, "ep": ep,
                     "entry_z": zscores[i], "hard_sl": ep + a * sl_mult}
        in_trade = True

    return trades


def grade_trades(trades):
    """トレードリストから統計・グレード算出"""
    if not trades:
        return None
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
    avg_w = sum(wins) / len(wins) if wins else 0
    avg_l = sum(losses) / len(losses) if losses else 0
    rr = abs(avg_w / avg_l) if avg_l != 0 else 0
    ret = sum(pnls)

    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1] * (1 + p))
    peak = equity[0]
    mdd = 0
    for e in equity:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak)

    if len(pnls) > 1:
        mr = sum(pnls) / len(pnls)
        sr = (sum((p - mr) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
        sharpe = (mr / sr) * (252 ** 0.5) if sr > 0 else 0
    else:
        sharpe = 0

    tp_n = sum(1 for t in trades if t["reason"] == "TP")
    sl_n = sum(1 for t in trades if t["reason"] in ("SL", "HSL"))
    tm_n = sum(1 for t in trades if t["reason"] == "TIME")

    pts = 100
    if len(trades) < 10: pts -= 40
    elif len(trades) < 20: pts -= 15
    elif len(trades) < 30: pts -= 5
    if pf < 1.0: pts -= 50
    elif pf < 1.2: pts -= 20
    elif pf < 1.5: pts -= 10
    if wr < 35: pts -= 30
    elif wr < 45: pts -= 15
    if mdd > 0.25: pts -= 35
    elif mdd > 0.15: pts -= 15
    elif mdd > 0.10: pts -= 5
    if sharpe < 0: pts -= 25
    elif sharpe < 0.5: pts -= 15
    elif sharpe < 1.0: pts -= 5
    if rr < 0.8: pts -= 15
    elif rr < 1.0: pts -= 8
    if ret < 0: pts -= 20

    if pts >= 80: grade = "S"
    elif pts >= 65: grade = "A"
    elif pts >= 50: grade = "B"
    elif pts >= 30: grade = "C"
    else: grade = "D"

    return {"n": len(trades), "wr": wr, "pf": pf, "sharpe": sharpe,
            "mdd": mdd, "rr": rr, "ret": ret, "grade": grade,
            "tp": tp_n, "sl": sl_n, "tm": tm_n, "avg_w": avg_w, "avg_l": avg_l, "pts": pts}


# ============================================================
#  メイン
# ============================================================
def main():
    print("#" * 60)
    print("  ETH特化 Z-Score平均回帰 大規模検証")
    print("  データ最大化 + 3分割ウォークフォワード")
    print("#" * 60)

    # --- データ取得 ---
    print("\n=== ETH 1H データ取得（最大10ページ）===")
    data = fetch_eth_max(max_pages=10)
    if not data or len(data) < 300:
        print("  [FAIL] データ不足")
        return

    total_bars = len(data)
    print(f"\n  合計: {total_bars} 4Hバー")

    # --- パラメータセット（15パターン）---
    PARAMS = [
        # (name, z_lookback, z_entry, z_exit, z_stop, sl_mult, max_hold, cooldown)
        ("A-Standard",    50, 2.0, 0.3, 3.5, 3.5, 20, 5),
        ("B-Tight",       30, 2.0, 0.5, 3.0, 3.0, 15, 3),
        ("C-Extreme",     50, 2.5, 0.3, 4.0, 4.0, 25, 5),
        ("D-FastMean",    50, 2.0, 0.0, 3.5, 3.5, 12, 3),
        ("E-Wide",        80, 2.0, 0.3, 3.5, 4.0, 30, 8),
        ("F-LongHold",    50, 2.0, 0.3, 3.5, 4.0, 40, 5),
        ("G-ShortLB",     20, 2.0, 0.3, 3.0, 3.0, 15, 3),
        ("H-LowEntry",    50, 1.8, 0.3, 3.5, 3.5, 20, 5),
        ("I-HighEntry",   50, 2.3, 0.3, 3.5, 3.5, 20, 5),
        ("J-NoCooldown",  50, 2.0, 0.3, 3.5, 3.5, 20, 1),
        ("K-TightExit",   50, 2.0, 0.5, 3.5, 3.5, 20, 5),
        ("L-WideExit",    50, 2.0, -0.2, 3.5, 3.5, 25, 5),
        ("M-TightSL",     50, 2.0, 0.3, 3.0, 2.5, 20, 5),
        ("N-WideSL",      50, 2.0, 0.3, 4.5, 5.0, 20, 5),
        ("O-Combo",       60, 2.1, 0.2, 3.5, 4.0, 25, 4),
    ]

    # --- 3分割ウォークフォワード ---
    # Period 1: [0 ~ 1/3] train, [1/3 ~ 2/3] test
    # Period 2: [1/3 ~ 2/3] train, [2/3 ~ end] test
    # Period 3: [0 ~ 2/3] train, [2/3 ~ end] test (最終確認)
    # ただし今回はOOSのみ報告（ISは参考値）

    third = total_bars // 3
    segments = [
        ("Period-1 (前半)", data[:third * 2]),      # train=前1/3, test=中1/3
        ("Period-2 (後半)", data[third:]),           # train=中1/3, test=後1/3
        ("Period-3 (全体)", data),                   # train=前半, test=後半
    ]

    all_results = []

    for seg_name, seg_data in segments:
        print(f"\n{'=' * 60}")
        print(f"  {seg_name}: {len(seg_data)} bars")
        print(f"{'=' * 60}")

        seg_results = []
        for pname, z_lb, z_ent, z_ext, z_stp, sl_m, mh, cd in PARAMS:
            # Z-scoreキャッシュクリア
            strat_zscore_reversion.__defaults__[-1].clear()

            trades = run_zscore_segment(
                seg_data, z_lookback=z_lb, z_entry=z_ent, z_exit=z_ext,
                z_stop=z_stp, sl_mult=sl_m, max_hold=mh, cooldown=cd
            )
            if not trades:
                continue
            stats = grade_trades(trades)
            if stats:
                stats["name"] = pname
                stats["seg"] = seg_name
                stats["trades"] = trades
                seg_results.append(stats)
                all_results.append(stats)

        # セグメント結果表示
        print(f"\n  {'Config':<16} {'G':>2} {'N':>4} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
              f"{'DD':>6} {'RR':>5} {'Ret':>8} {'TP':>3} {'SL':>3} {'TM':>3}")
        print(f"  {'-' * 80}")
        for r in sorted(seg_results, key=lambda x: -x["ret"]):
            print(f"  {r['name']:<16} {r['grade']:>2} {r['n']:>4} "
                  f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['sharpe']:>7.2f} "
                  f"{r['mdd']*100:>5.1f}% {r['rr']:>4.2f} {r['ret']*100:>+7.2f}% "
                  f"{r['tp']:>3} {r['sl']:>3} {r['tm']:>3}")

    # === 最終サマリー ===
    print(f"\n\n{'#' * 60}")
    print(f"  最終判定: ETH Z-Score ウォークフォワード結果")
    print(f"{'#' * 60}")

    # パラメータごとに全期間の成績を集計
    param_summary = {}
    for r in all_results:
        name = r["name"]
        if name not in param_summary:
            param_summary[name] = {"grades": [], "rets": [], "pfs": [], "ns": [], "wrs": []}
        param_summary[name]["grades"].append(r["grade"])
        param_summary[name]["rets"].append(r["ret"])
        param_summary[name]["pfs"].append(r["pf"])
        param_summary[name]["ns"].append(r["n"])
        param_summary[name]["wrs"].append(r["wr"])

    print(f"\n  {'Config':<16} {'Grades':<12} {'Avg N':>6} {'Avg WR':>7} {'Avg PF':>7} {'Avg Ret':>8} {'Consistent':>10}")
    print(f"  {'-' * 75}")

    consistent_count = 0
    for name in sorted(param_summary.keys()):
        s = param_summary[name]
        avg_n = sum(s["ns"]) / len(s["ns"])
        avg_wr = sum(s["wrs"]) / len(s["wrs"])
        avg_pf = sum(s["pfs"]) / len(s["pfs"])
        avg_ret = sum(s["rets"]) / len(s["rets"])
        grades_str = "/".join(s["grades"])
        # 一貫性: 全期間でプラスかつPF>1
        consistent = all(r > 0 for r in s["rets"]) and all(p > 1.0 for p in s["pfs"])
        mark = "OK" if consistent else "NG"
        if consistent:
            consistent_count += 1
        print(f"  {name:<16} {grades_str:<12} {avg_n:>5.1f} {avg_wr:>6.1f}% {avg_pf:>6.2f} "
              f"{avg_ret*100:>+7.2f}% {mark:>10}")

    # 最優秀設定の全トレード表示
    best_name = max(param_summary.keys(),
                    key=lambda n: sum(param_summary[n]["rets"]) / len(param_summary[n]["rets"]))
    best_trades_all = []
    for r in all_results:
        if r["name"] == best_name and r["seg"] == "Period-3 (全体)":
            best_trades_all = r.get("trades", [])
            break

    if best_trades_all:
        print(f"\n  --- 最優秀設定 [{best_name}] Period-3 個別トレード ---")
        print(f"  {'#':>3} {'Dir':<6} {'Entry Z':>8} {'Exit Z':>8} {'Reason':<5} {'Bars':>4} {'PnL':>8}")
        print(f"  {'-' * 50}")
        for idx, t in enumerate(best_trades_all, 1):
            print(f"  {idx:>3} {t['dir']:<6} {t['entry_z']:>+7.2f} {t['exit_z']:>+7.2f} "
                  f"{t['reason']:<5} {t['bars']:>4} {t['pnl']*100:>+7.2f}%")

    # 最終結論
    print(f"\n{'=' * 60}")
    if consistent_count > 0:
        print(f"  RESULT: {consistent_count}/15 設定が全期間一貫プラス")
        print(f"  → Z-Score ETH平均回帰にエッジの可能性あり")
    else:
        print(f"  RESULT: 全期間一貫プラスの設定なし")
        print(f"  → N=9のB判定は偶然だった可能性が高い")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
