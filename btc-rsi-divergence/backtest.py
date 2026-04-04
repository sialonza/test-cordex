#!/usr/bin/env python3
"""
Big Move Predictor Backtest Framework
Pine Script のコアロジックをPythonで再実装し、実データで検証する。
Usage: python3 backtest.py
"""
import math
import json
import urllib.request
from datetime import datetime

# ============================================================
#  データ取得（Crypto.com public API）
# ============================================================
def fetch_ohlcv(symbol="BTC_USDT", timeframe="4h"):
    """Crypto.com public candlestick API"""
    tf_map = {"1h": "1h", "4h": "4h", "1D": "1D"}
    tf = tf_map.get(timeframe, "4h")
    url = f"https://api.crypto.com/exchange/v1/public/get-candlestick?instrument_name={symbol}&timeframe={tf}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            candles = data.get("result", {}).get("data", [])
            rows = []
            seen = set()
            for c in candles:
                ts = c["t"] if "t" in c else 0
                if ts in seen:
                    continue
                seen.add(ts)
                rows.append({
                    "ts": ts,
                    "o": float(c.get("o", 0)),
                    "h": float(c.get("h", 0)),
                    "l": float(c.get("l", 0)),
                    "c": float(c.get("c", 0)),
                    "v": float(c.get("v", 0)),
                })
            rows.sort(key=lambda x: x["ts"])
            return rows
    except Exception as e:
        print(f"[WARN] API fetch failed: {e}, using embedded sample data")
        return None


def generate_sample_data(n=500):
    """ランダムウォークでBTCライクなOHLCVを生成"""
    import random
    random.seed(42)
    price = 65000.0
    data = []
    for i in range(n):
        vol = random.uniform(100, 2000)
        # レジーム切替: 圧縮→爆発を繰り返す
        cycle = (i % 80)
        if cycle < 50:  # 圧縮局面
            move = random.gauss(0, price * 0.003)
        else:  # 爆発局面
            move = random.gauss(0, price * 0.012)
            if cycle == 50:
                move = random.choice([-1, 1]) * price * 0.025
        o = price
        c = price + move
        h = max(o, c) + abs(random.gauss(0, price * 0.002))
        l = min(o, c) - abs(random.gauss(0, price * 0.002))
        price = c
        data.append({"ts": i, "o": o, "h": h, "l": l, "c": c, "v": vol})
    return data


# ============================================================
#  テクニカル指標
# ============================================================
def calc_ema(src, period):
    if not src:
        return []
    e = [src[0]]
    m = 2.0 / (period + 1)
    for i in range(1, len(src)):
        e.append(src[i] * m + e[-1] * (1 - m))
    return e


def calc_rsi(closes, period=14):
    rsi = [50.0] * min(period, len(closes))
    if len(closes) <= period:
        return rsi
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    avg_g = gains / period
    avg_l = losses / period
    for i in range(period, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0)) / period
        rs = avg_g / avg_l if avg_l > 0 else 100
        rsi.append(100 - 100 / (1 + rs))
    return rsi


def calc_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(len(highs)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
    atr = [0.0] * min(period, len(trs))
    if len(trs) >= period:
        a = sum(trs[:period]) / period
        atr = [0.0] * (period - 1) + [a]
        for i in range(period, len(trs)):
            a = (a * (period - 1) + trs[i]) / period
            atr.append(a)
    return atr


def calc_bb(closes, period=20, mult=2.0):
    upper, basis, lower = [], [], []
    for i in range(len(closes)):
        start = max(0, i - period + 1)
        w = closes[start:i + 1]
        m = sum(w) / len(w)
        std = (sum((x - m) ** 2 for x in w) / len(w)) ** 0.5
        basis.append(m)
        upper.append(m + mult * std)
        lower.append(m - mult * std)
    return upper, basis, lower


def calc_kc(closes, highs, lows, period=20, mult=1.5):
    mid = calc_ema(closes, period)
    atr_kc = calc_atr(highs, lows, closes, period)
    upper = [mid[i] + atr_kc[i] * mult for i in range(len(mid))]
    lower = [mid[i] - atr_kc[i] * mult for i in range(len(mid))]
    return upper, mid, lower


# ============================================================
#  スコア計算（14因子）
# ============================================================
def calc_squeeze_score(i, closes, highs, lows, opens, vols,
                       bbU, bbB, bbL, kcU, kcM, kcL,
                       atr, rsi, ema9, ema21, volSMA):
    sqzLen = 20
    score = 0.0

    # (1) BB Squeeze
    bbWidth = (bbU[i] - bbL[i]) / bbB[i] * 100 if bbB[i] > 0 else 0
    s, e = max(0, i - sqzLen * 2 + 1), i + 1
    bbWidths = [(bbU[j] - bbL[j]) / bbB[j] * 100 if bbB[j] > 0 else 0 for j in range(s, e)]
    bbWidthMin = min(bbWidths) if bbWidths else 0
    bbWidthMax = max(bbWidths) if bbWidths else 0
    bbComp = 1.0 - (bbWidth - bbWidthMin) / (bbWidthMax - bbWidthMin) if bbWidthMax > bbWidthMin else 0
    bbSma = sum(bbWidths[-sqzLen:]) / min(sqzLen, len(bbWidths)) if bbWidths else 0
    bbSqueeze = bbWidth < bbSma * 0.75
    score += 7.0 if bbSqueeze else 0
    score += bbComp * 5.0

    # (2) KC Squeeze
    kcSqueeze = bbU[i] < kcU[i] and bbL[i] > kcL[i]
    score += 15.0 if kcSqueeze else 0

    # (3) ATR圧縮
    atrW = atr[max(0, i - sqzLen + 1):i + 1]
    atrComp = 1.0 - (atr[i] - min(atrW)) / (max(atrW) - min(atrW)) if max(atrW) > min(atrW) else 0
    score += atrComp * 10.0

    # (4) 出来高枯渇
    vr = vols[i] / volSMA[i] if volSMA[i] > 0 else 1
    score += 4.0 if vr < 0.6 else 0
    decay = sum(1 for j in range(5) if i - j >= 0 and vols[i - j] < volSMA[i - j])
    score += min(4.0, decay * 0.8)

    # (5) Narrow Range
    ranges = [highs[j] - lows[j] for j in range(max(0, i - sqzLen + 1), i + 1)]
    rangeSMA = sum(ranges) / len(ranges) if ranges else 1
    nrCount = sum(1 for j in range(4) if i - j >= 0 and rangeSMA > 0 and (highs[i - j] - lows[i - j]) < rangeSMA * 0.6)
    score += min(8.0, nrCount * 2.0)

    # (6) Inside Bar
    ibCount = sum(1 for j in range(5) if i - j >= 1 and highs[i - j] < highs[i - j - 1] and lows[i - j] > lows[i - j - 1])
    score += min(6.0, ibCount * 1.5)

    # (7) RSI圧縮
    rsiW = rsi[max(0, i - sqzLen + 1):i + 1]
    rsiRange = max(rsiW) - min(rsiW) if rsiW else 30
    score += 6.0 if rsiRange < 15 else 0

    # (8) EMA収束
    emaDiff = abs(ema9[i] - ema21[i]) / closes[i] * 100 if closes[i] > 0 else 0
    score += 4.0 if emaDiff < 0.3 else 0

    # (9) ローソク足パターン
    bodySize = abs(closes[i] - opens[i])
    candleRange = highs[i] - lows[i]
    if candleRange > 0:
        isDoji = bodySize / candleRange < 0.1
        if i > 0:
            bullEngulf = closes[i] > opens[i] and closes[i - 1] < opens[i - 1] and closes[i] > opens[i - 1] and opens[i] < closes[i - 1]
            bearEngulf = closes[i] < opens[i] and closes[i - 1] > opens[i - 1] and closes[i] < opens[i - 1] and opens[i] > closes[i - 1]
        else:
            bullEngulf = bearEngulf = False
        if isDoji or bullEngulf or bearEngulf:
            score += 5.0

    finalScore = min(100.0, round(score / 1.18))
    return finalScore, kcSqueeze, bbSqueeze, atrComp


# ============================================================
#  方向予測（22因子・簡略版）
# ============================================================
def calc_direction(i, closes, highs, lows, opens, vols,
                   rsi, ema9, ema21, atr, bbU, bbL, volSMA):
    bullW = 0.0
    bearW = 0.0

    # D1 RSI
    if rsi[i] < 25: bullW += 5
    elif rsi[i] < 30: bullW += 3.5
    elif rsi[i] < 40: bullW += 1.5
    if rsi[i] > 75: bearW += 5
    elif rsi[i] > 70: bearW += 3.5
    elif rsi[i] > 60: bearW += 1.5

    # D2 RSI momentum
    if i >= 3:
        rm = rsi[i] - rsi[i - 3]
        if rm > 8: bullW += 3
        elif rm > 4: bullW += 1.5
        if rm < -8: bearW += 3
        elif rm < -4: bearW += 1.5

    # D3 EMA alignment
    if ema9[i] > ema21[i]: bullW += 3
    else: bearW += 3

    # D5 MACD (simplified as ema9-ema21)
    macd = ema9[i] - ema21[i]
    if macd > 0: bullW += 1.5
    else: bearW += 1.5
    if i > 0:
        macd_prev = ema9[i - 1] - ema21[i - 1]
        if macd > 0 and macd > macd_prev: bullW += 2.5
        if macd < 0 and macd < macd_prev: bearW += 2.5

    # D7 OBV trend (simplified)
    if i >= 5:
        obv = 0
        for j in range(max(0, i - 20), i + 1):
            if j > 0:
                if closes[j] > closes[j - 1]: obv += vols[j]
                elif closes[j] < closes[j - 1]: obv -= vols[j]
        if obv > 0: bullW += 2
        else: bearW += 2

    # D8 Smart money
    if i >= 3:
        priceUp = closes[i] > closes[i - 3]
        volUp = vols[i] > volSMA[i] if volSMA[i] > 0 else False
        if priceUp and volUp: bullW += 3
        elif priceUp and not volUp: bearW += 1
        if not priceUp and volUp: bearW += 3
        elif not priceUp and not volUp: bullW += 1

    # D10 Squeeze momentum
    if i >= 20:
        sqzMid = (max(highs[i - 19:i + 1]) + min(lows[i - 19:i + 1])) / 2
        sqzMom = closes[i] - sqzMid
        if sqzMom > 0: bullW += 3
        else: bearW += 3

    # D11 Market structure
    if i >= 20:
        swH1 = max(highs[max(0, i - 9):i + 1])
        swL1 = min(lows[max(0, i - 9):i + 1])
        swH2 = max(highs[max(0, i - 19):max(0, i - 9)])
        swL2 = min(lows[max(0, i - 19):max(0, i - 9)])
        if swH1 > swH2 and swL1 > swL2: bullW += 4
        elif swL1 > swL2: bullW += 2
        if swH1 < swH2 and swL1 < swL2: bearW += 4
        elif swH1 < swH2: bearW += 2

    # D12 ROC
    if i >= 10:
        roc = (closes[i] - closes[i - 10]) / closes[i - 10] * 100
        if roc > 3: bullW += 2
        elif roc > 0: bullW += 0.5
        if roc < -3: bearW += 2
        elif roc < 0: bearW += 0.5

    # D16 Engulfing
    if i > 0:
        if closes[i] > opens[i] and closes[i - 1] < opens[i - 1] and closes[i] > opens[i - 1] and opens[i] < closes[i - 1]:
            bullW += 3
        if closes[i] < opens[i] and closes[i - 1] > opens[i - 1] and closes[i] < opens[i - 1] and opens[i] > closes[i - 1]:
            bearW += 3

    # D18 BB position
    bbRange = bbU[i] - bbL[i]
    if bbRange > 0:
        bbPos = (closes[i] - bbL[i]) / bbRange
        if bbPos < 0.1: bullW += 2
        elif bbPos < 0.3: bullW += 1
        if bbPos > 0.9: bearW += 2
        elif bbPos > 0.7: bearW += 1

    totalDir = bullW + bearW
    dirProb = round(max(bullW, bearW) / totalDir * 100) if totalDir > 0 else 50
    isBull = bullW > bearW
    return isBull, dirProb, bullW, bearW


# ============================================================
#  バックテスト実行
# ============================================================
def run_backtest(data, min_score=45, confirm_bars=5, slippage=0.001, commission=0.001):
    n = len(data)
    if n < 50:
        print("[ERROR] Not enough data (need 50+ bars)")
        return

    closes = [d["c"] for d in data]
    highs = [d["h"] for d in data]
    lows = [d["l"] for d in data]
    opens = [d["o"] for d in data]
    vols = [d["v"] for d in data]

    # Indicators
    rsi = calc_rsi(closes)
    atr = calc_atr(highs, lows, closes)
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    bbU, bbB, bbL = calc_bb(closes)
    kcU, kcM, kcL = calc_kc(closes, highs, lows)
    volSMA = calc_ema(vols, 20)

    # Walk-forward: train 70%, test 30%
    split = int(n * 0.7)
    test_start = max(split, 25)

    # Stats
    total_signals = 0
    hit_signals = 0  # ATR*2 move within confirm_bars
    dir_signals = 0
    dir_hits = 0
    t1_hits = 0
    t2_hits = 0
    trades_pnl = []
    last_sig_bar = -10

    # Factor importance tracking
    factor_counts = {}

    for i in range(test_start, n - confirm_bars):
        score, kcSqz, bbSqz, atrComp = calc_squeeze_score(
            i, closes, highs, lows, opens, vols,
            bbU, bbB, bbL, kcU, kcM, kcL, atr, rsi, ema9, ema21, volSMA)

        if score < min_score:
            continue
        if (i - last_sig_bar) < 3:
            continue
        last_sig_bar = i

        isBull, dirProb, bullW, bearW = calc_direction(
            i, closes, highs, lows, opens, vols,
            rsi, ema9, ema21, atr, bbU, bbL, volSMA)

        total_signals += 1

        # Check outcome in next confirm_bars
        maxUp = 0
        maxDn = 0
        for j in range(1, confirm_bars + 1):
            if i + j < n:
                maxUp = max(maxUp, highs[i + j] - closes[i])
                maxDn = max(maxDn, closes[i] - lows[i + j])

        maxMove = max(maxUp, maxDn)
        hit = maxMove >= atr[i] * 1.5  # relaxed for 4H
        if hit:
            hit_signals += 1

        # Direction accuracy
        if dirProb >= 60:
            dir_signals += 1
            if isBull and maxUp > maxDn:
                dir_hits += 1
            elif not isBull and maxDn > maxUp:
                dir_hits += 1

        # Target accuracy
        energy = (atrComp + (1 if kcSqz else 0)) / 2
        eMult = 1.5 + energy * 2.5
        expMove = atr[i] * eMult
        t1Move = expMove * 0.6
        t2Move = expMove * 1.0

        if isBull:
            if maxUp >= t1Move: t1_hits += 1
            if maxUp >= t2Move: t2_hits += 1
        else:
            if maxDn >= t1Move: t1_hits += 1
            if maxDn >= t2Move: t2_hits += 1

        # Simulated trade PnL
        entry = closes[i] * (1 + slippage)  # slippage
        if i + confirm_bars < n:
            exit_price = closes[i + confirm_bars]
            if isBull:
                pnl = (exit_price - entry) / entry - commission * 2
            else:
                pnl = (entry - exit_price) / entry - commission * 2
            trades_pnl.append(pnl)

    # ============================================================
    #  結果出力
    # ============================================================
    print("=" * 60)
    print("  Big Move Predictor バックテスト結果")
    print("=" * 60)
    print(f"  データ本数:     {n} bars")
    print(f"  テスト期間:     bar {test_start} ~ {n - confirm_bars}")
    print(f"  最小スコア:     {min_score}%")
    print(f"  確認期間:       {confirm_bars} bars")
    print(f"  スリッページ:   {slippage * 100:.1f}%")
    print(f"  手数料:         {commission * 100:.1f}% (往復)")

    print(f"\n--- シグナル精度 ---")
    hit_rate = round(hit_signals / total_signals * 100) if total_signals > 0 else 0
    print(f"  総シグナル:     {total_signals}")
    print(f"  大変動的中:     {hit_signals}/{total_signals} = {hit_rate}%")

    dir_rate = round(dir_hits / dir_signals * 100) if dir_signals > 0 else 0
    print(f"  方向予測数:     {dir_signals}")
    print(f"  方向的中:       {dir_hits}/{dir_signals} = {dir_rate}%")

    t1_rate = round(t1_hits / total_signals * 100) if total_signals > 0 else 0
    t2_rate = round(t2_hits / total_signals * 100) if total_signals > 0 else 0
    print(f"  T1到達率:       {t1_hits}/{total_signals} = {t1_rate}%")
    print(f"  T2到達率:       {t2_hits}/{total_signals} = {t2_rate}%")

    # Trade stats
    if trades_pnl:
        wins = [p for p in trades_pnl if p > 0]
        losses = [p for p in trades_pnl if p <= 0]
        total_return = sum(trades_pnl)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        win_rate = len(wins) / len(trades_pnl) * 100
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

        # Max drawdown
        equity = [1.0]
        for p in trades_pnl:
            equity.append(equity[-1] * (1 + p))
        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)

        # Sharpe (annualized, assume 6 trades/day for 4H)
        if len(trades_pnl) > 1:
            mean_r = sum(trades_pnl) / len(trades_pnl)
            std_r = (sum((p - mean_r) ** 2 for p in trades_pnl) / (len(trades_pnl) - 1)) ** 0.5
            sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
        else:
            sharpe = 0

        print(f"\n--- トレード統計 ---")
        print(f"  トレード数:     {len(trades_pnl)}")
        print(f"  勝率:           {win_rate:.1f}%")
        print(f"  平均利益:       {avg_win * 100:.2f}%")
        print(f"  平均損失:       {avg_loss * 100:.2f}%")
        print(f"  Profit Factor:  {profit_factor:.2f}")
        print(f"  累積リターン:   {total_return * 100:.2f}%")
        print(f"  最大DD:         {max_dd * 100:.2f}%")
        print(f"  Sharpe Ratio:   {sharpe:.2f}")
        print(f"  最終エクイティ: {equity[-1]:.4f}")

    # Verdict
    print(f"\n{'=' * 60}")
    print(f"  実戦判定")
    print(f"{'=' * 60}")
    if hit_rate >= 60 and dir_rate >= 55 and (not trades_pnl or sharpe > 0.5):
        print(f"  ✓ 実用圏内（Hit:{hit_rate}% Dir:{dir_rate}%）")
    elif hit_rate >= 50:
        print(f"  △ 条件付き使用可（Hit:{hit_rate}% Dir:{dir_rate}%）")
        print(f"    → S級シグナルのみ使用、SL必須")
    else:
        print(f"  ✗ 要改善（Hit:{hit_rate}% Dir:{dir_rate}%）")
        print(f"    → 閾値調整 or 因子削減が必要")


# ============================================================
#  メイン
# ============================================================
if __name__ == "__main__":
    print("BTC/USDT 4H データ取得中...")
    data = fetch_ohlcv("BTC_USDT", "4h")

    if data and len(data) >= 50:
        print(f"API: {len(data)} bars 取得")
        run_backtest(data, min_score=40, confirm_bars=5)
    else:
        print("APIデータ不足、サンプルデータで実行...")
        data = generate_sample_data(500)
        print(f"サンプル: {len(data)} bars 生成")
        run_backtest(data, min_score=40, confirm_bars=5)
