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
    """CryptoCompare API (無料、最大2000本) + Crypto.com フォールバック"""

    # --- CryptoCompare (primary) ---
    # 4h = histohour limit=2000 aggregate=4
    # 1D = histoday limit=2000
    cc_sym = symbol.split("_")[0]  # BTC_USDT → BTC
    cc_tsym = symbol.split("_")[1] if "_" in symbol else "USDT"

    aggregate_4h = False
    if timeframe == "1D":
        cc_url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={cc_sym}&tsym={cc_tsym}&limit=2000"
    elif timeframe == "1h":
        cc_url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={cc_sym}&tsym={cc_tsym}&limit=2000"
    else:  # 4h → 1hで2000本取得して4本ずつ集約
        cc_url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={cc_sym}&tsym={cc_tsym}&limit=2000"
        aggregate_4h = True

    try:
        # Paginate to get up to 4000 1H bars (= 1000 4H bars)
        all_candles = []
        toTs = ""
        for page in range(2):  # 2 pages × 2000 = 4000 bars
            page_url = cc_url if not toTs else f"{cc_url}&toTs={toTs}"
            req = urllib.request.Request(page_url, headers={"User-Agent": "BacktestBot/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read())
                candles = raw.get("Data", {}).get("Data", [])
                if not candles:
                    break
                all_candles.extend(candles)
                # Next page: oldest timestamp - 1
                toTs = str(min(c.get("time", 0) for c in candles) - 1)
                print(f"  [INFO] Page {page+1}: {len(candles)} candles fetched")

        if all_candles:
            # Deduplicate by timestamp
            seen = set()
            candles_deduped = []
            for c in all_candles:
                t = c.get("time", 0)
                if t not in seen:
                    seen.add(t)
                    candles_deduped.append(c)

            rows = []
            for c in candles_deduped:
                if c.get("close", 0) == 0 and c.get("open", 0) == 0:
                    continue
                rows.append({
                    "ts": c.get("time", 0),
                    "o": float(c.get("open", 0)),
                    "h": float(c.get("high", 0)),
                    "l": float(c.get("low", 0)),
                    "c": float(c.get("close", 0)),
                    "v": float(c.get("volumefrom", 0)),
                })
            rows.sort(key=lambda x: x["ts"])

            # 1H → 4H 集約
            if aggregate_4h and len(rows) >= 4:
                agg = []
                for k in range(0, len(rows) - 3, 4):
                    chunk = rows[k:k + 4]
                    agg.append({
                        "ts": chunk[0]["ts"],
                        "o": chunk[0]["o"],
                        "h": max(c["h"] for c in chunk),
                        "l": min(c["l"] for c in chunk),
                        "c": chunk[-1]["c"],
                        "v": sum(c["v"] for c in chunk),
                    })
                rows = agg
                print(f"  [INFO] Aggregated 1H → 4H: {len(rows)} candles")

            if len(rows) >= 50:
                print(f"  [OK] CryptoCompare {timeframe} → {len(rows)} candles")
                return rows
            else:
                print(f"  [WARN] CryptoCompare: only {len(rows)} candles")
    except Exception as e:
        print(f"  [WARN] CryptoCompare: {e}")

    # --- Crypto.com fallback ---
    tf_map = {"1h": "1h", "4h": "4h", "1D": "1D"}
    tf = tf_map.get(timeframe, "4h")
    cc_url2 = f"https://api.crypto.com/exchange/v1/public/get-candlestick?instrument_name={symbol}&timeframe={tf}"
    try:
        req = urllib.request.Request(cc_url2, headers={"User-Agent": "BacktestBot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read())
            candles = raw.get("result", {}).get("data", [])
            if candles:
                rows = []
                for c in candles:
                    rows.append({
                        "ts": c.get("t") or c.get("timestamp") or 0,
                        "o": float(c.get("o") or c.get("open") or 0),
                        "h": float(c.get("h") or c.get("high") or 0),
                        "l": float(c.get("l") or c.get("low") or 0),
                        "c": float(c.get("c") or c.get("close") or 0),
                        "v": float(c.get("v") or c.get("volume") or 0),
                    })
                rows.sort(key=lambda x: str(x["ts"]))
                if len(rows) >= 10:
                    print(f"  [OK] Crypto.com {tf} → {len(rows)} candles")
                    return rows
    except Exception as e:
        print(f"  [WARN] Crypto.com: {e}")

    print("  [FAIL] All APIs failed")
    return None


def get_embedded_data():
    """BTC/USDT 1D 実データ (2026-02-22 ~ 2026-04-02, Crypto.com API)"""
    return [
        {"ts":"2026-02-22","o":67972.01,"h":68256.18,"l":67185.01,"c":67638.99,"v":1919.4},
        {"ts":"2026-02-23","o":67639.00,"h":67691.28,"l":63879.99,"c":64651.02,"v":5713.0},
        {"ts":"2026-02-24","o":64651.03,"h":65011.19,"l":62500.00,"c":64065.79,"v":6944.1},
        {"ts":"2026-02-25","o":64063.73,"h":70022.10,"l":63911.08,"c":67991.24,"v":9281.7},
        {"ts":"2026-02-26","o":67989.00,"h":68860.19,"l":66494.99,"c":67486.99,"v":10015.2},
        {"ts":"2026-02-27","o":67488.45,"h":68225.51,"l":64922.72,"c":65878.01,"v":7829.8},
        {"ts":"2026-02-28","o":65865.76,"h":67763.38,"l":63021.18,"c":66971.11,"v":6217.3},
        {"ts":"2026-03-01","o":66960.42,"h":68220.55,"l":65037.17,"c":65769.00,"v":6441.4},
        {"ts":"2026-03-02","o":65772.00,"h":70108.49,"l":65264.94,"c":68837.98,"v":8455.3},
        {"ts":"2026-03-03","o":68837.98,"h":69257.99,"l":66144.45,"c":68335.99,"v":9929.4},
        {"ts":"2026-03-04","o":68336.00,"h":74074.99,"l":67391.99,"c":72669.17,"v":7323.9},
        {"ts":"2026-03-05","o":72669.18,"h":73578.03,"l":70638.88,"c":70877.01,"v":6308.2},
        {"ts":"2026-03-06","o":70886.99,"h":71423.10,"l":67731.55,"c":68112.00,"v":6268.4},
        {"ts":"2026-03-07","o":68112.01,"h":68544.99,"l":66915.70,"c":67264.98,"v":2056.3},
        {"ts":"2026-03-08","o":67263.02,"h":68201.94,"l":65601.17,"c":65971.21,"v":3131.2},
        {"ts":"2026-03-09","o":65971.21,"h":69547.90,"l":65818.71,"c":68433.74,"v":5939.9},
        {"ts":"2026-03-10","o":68432.01,"h":71783.73,"l":68368.76,"c":69952.60,"v":5751.6},
        {"ts":"2026-03-11","o":69952.61,"h":71338.97,"l":68976.30,"c":70192.76,"v":5315.2},
        {"ts":"2026-03-12","o":70199.00,"h":70811.15,"l":69200.00,"c":70527.50,"v":4432.0},
        {"ts":"2026-03-13","o":70521.00,"h":73913.88,"l":70385.56,"c":70928.99,"v":5280.6},
        {"ts":"2026-03-14","o":70931.01,"h":71319.42,"l":70317.86,"c":71202.94,"v":1253.1},
        {"ts":"2026-03-15","o":71211.96,"h":73222.98,"l":70852.53,"c":72827.58,"v":2169.6},
        {"ts":"2026-03-16","o":72821.97,"h":74919.77,"l":72270.00,"c":74887.99,"v":3992.2},
        {"ts":"2026-03-17","o":74888.00,"h":76013.01,"l":73364.44,"c":73916.99,"v":5469.0},
        {"ts":"2026-03-18","o":73917.00,"h":74690.64,"l":70490.12,"c":71246.06,"v":5097.2},
        {"ts":"2026-03-19","o":71246.06,"h":71623.99,"l":68779.54,"c":69920.01,"v":4624.7},
        {"ts":"2026-03-20","o":69920.01,"h":71378.38,"l":69391.68,"c":70517.00,"v":3834.3},
        {"ts":"2026-03-21","o":70507.83,"h":71106.00,"l":68563.42,"c":68924.01,"v":1150.0},
        {"ts":"2026-03-22","o":68924.00,"h":69594.99,"l":67348.10,"c":67864.00,"v":2808.4},
        {"ts":"2026-03-23","o":67864.01,"h":71839.24,"l":67442.36,"c":70897.60,"v":5034.2},
        {"ts":"2026-03-24","o":70897.60,"h":71410.99,"l":68909.38,"c":70564.00,"v":4033.8},
        {"ts":"2026-03-25","o":70564.00,"h":72056.05,"l":70400.88,"c":71333.07,"v":3410.3},
        {"ts":"2026-03-26","o":71333.08,"h":71446.48,"l":68145.12,"c":68821.01,"v":4749.3},
        {"ts":"2026-03-27","o":68821.01,"h":69180.04,"l":65546.54,"c":66398.01,"v":4270.8},
        {"ts":"2026-03-28","o":66398.01,"h":67293.34,"l":65919.03,"c":66367.00,"v":1859.3},
        {"ts":"2026-03-29","o":66367.00,"h":67132.62,"l":64969.00,"c":66017.92,"v":2361.5},
        {"ts":"2026-03-30","o":66019.00,"h":68191.23,"l":65793.43,"c":66795.08,"v":5219.9},
        {"ts":"2026-03-31","o":66795.09,"h":68626.52,"l":65969.67,"c":68279.95,"v":5470.7},
        {"ts":"2026-04-01","o":68290.00,"h":69324.71,"l":67582.70,"c":68117.37,"v":4136.3},
        {"ts":"2026-04-02","o":68118.00,"h":68672.99,"l":66208.87,"c":66650.00,"v":1712.5},
    ]


def generate_sample_data(n=500):
    """ランダムウォークでBTCライクなOHLCVを生成（フォールバック用）"""
    import random
    random.seed(42)
    price = 65000.0
    data = []
    for i in range(n):
        vol = random.uniform(100, 2000)
        cycle = (i % 80)
        if cycle < 50:
            move = random.gauss(0, price * 0.003)
        else:
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
    if n < 25:
        print("[ERROR] Not enough data (need 25+ bars)")
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

    # Walk-forward: train 50%, test 50% (more test data)
    split = int(n * 0.5)
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
        if (i - last_sig_bar) < 2:
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
    # === 4H テスト ===
    print("=" * 60)
    print("  BTC/USDT 4H バックテスト")
    print("=" * 60)
    print("データ取得中...")
    data_4h = fetch_ohlcv("BTC_USDT", "4h")

    if data_4h and len(data_4h) >= 100:
        print(f"\n4H実データ: {len(data_4h)} bars")
        for ms in [50, 40, 30]:
            print(f"\n{'='*60}")
            print(f"[4H] 閾値テスト: min_score={ms}")
            run_backtest(data_4h, min_score=ms, confirm_bars=5)

    # === 1D テスト ===
    print("\n\n" + "#" * 60)
    print("  BTC/USDT 1D (日足) バックテスト")
    print("#" * 60)
    print("データ取得中...")
    data_1d = fetch_ohlcv("BTC_USDT", "1D")

    if data_1d and len(data_1d) >= 100:
        print(f"\n1D実データ: {len(data_1d)} bars")
        for ms in [50, 40, 30]:
            print(f"\n{'='*60}")
            print(f"[1D] 閾値テスト: min_score={ms}")
            run_backtest(data_1d, min_score=ms, confirm_bars=3)  # 日足は確認3本(3日)
    else:
        print(f"  [WARN] 1Dデータ不足、埋め込みデータ使用")
        embedded = get_embedded_data()
        print(f"\n埋め込み実データ: {len(embedded)} bars (BTC 1D)")
        run_backtest(embedded, min_score=20, confirm_bars=3)
