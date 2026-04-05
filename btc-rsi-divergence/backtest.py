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
        for page in range(4):  # 4 pages × 2000 = 8000 bars max
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
                   rsi, ema9, ema21, atr, bbU, bbL, volSMA,
                   is_daily=False):
    bullW = 0.0
    bearW = 0.0

    # D1 RSI — 日足はレンジが狭いので閾値を緩和
    if is_daily:
        if rsi[i] < 35: bullW += 5
        elif rsi[i] < 45: bullW += 2.5
        if rsi[i] > 65: bearW += 5
        elif rsi[i] > 55: bearW += 2.5
    else:
        if rsi[i] < 25: bullW += 5
        elif rsi[i] < 30: bullW += 3.5
        elif rsi[i] < 40: bullW += 1.5
        if rsi[i] > 75: bearW += 5
        elif rsi[i] > 70: bearW += 3.5
        elif rsi[i] > 60: bearW += 1.5

    # D2 RSI momentum — 日足は5日間で見る
    lookback = 5 if is_daily else 3
    if i >= lookback:
        rm = rsi[i] - rsi[i - lookback]
        if rm > 8: bullW += 3
        elif rm > 4: bullW += 1.5
        if rm < -8: bearW += 3
        elif rm < -4: bearW += 1.5

    # D3 EMA alignment
    if ema9[i] > ema21[i]: bullW += 3
    else: bearW += 3

    # D3b EMA slope (日足用追加: EMAの傾き方向)
    if is_daily and i >= 5:
        ema21_slope = ema21[i] - ema21[i - 5]
        if ema21_slope > 0: bullW += 3
        else: bearW += 3

    # D5 MACD (simplified as ema9-ema21)
    macd = ema9[i] - ema21[i]
    if macd > 0: bullW += 1.5
    else: bearW += 1.5
    if i > 0:
        macd_prev = ema9[i - 1] - ema21[i - 1]
        if macd > 0 and macd > macd_prev: bullW += 2.5
        if macd < 0 and macd < macd_prev: bearW += 2.5

    # D7 OBV trend — 日足は40日で見る
    obv_period = 40 if is_daily else 20
    if i >= 5:
        obv = 0
        for j in range(max(0, i - obv_period), i + 1):
            if j > 0:
                if closes[j] > closes[j - 1]: obv += vols[j]
                elif closes[j] < closes[j - 1]: obv -= vols[j]
        if obv > 0: bullW += 2
        else: bearW += 2

    # D8 Smart money — 日足は5日前と比較
    sm_lb = 5 if is_daily else 3
    if i >= sm_lb:
        priceUp = closes[i] > closes[i - sm_lb]
        volUp = vols[i] > volSMA[i] if volSMA[i] > 0 else False
        if priceUp and volUp: bullW += 3
        elif priceUp and not volUp: bearW += 1
        if not priceUp and volUp: bearW += 3
        elif not priceUp and not volUp: bullW += 1

    # D10 Squeeze momentum — 日足は40日
    sqz_period = 40 if is_daily else 20
    if i >= sqz_period:
        sqzMid = (max(highs[i - sqz_period + 1:i + 1]) + min(lows[i - sqz_period + 1:i + 1])) / 2
        sqzMom = closes[i] - sqzMid
        if sqzMom > 0: bullW += 3
        else: bearW += 3

    # D11 Market structure — 日足は20/40日スイング
    if is_daily and i >= 40:
        swH1 = max(highs[max(0, i - 19):i + 1])
        swL1 = min(lows[max(0, i - 19):i + 1])
        swH2 = max(highs[max(0, i - 39):max(0, i - 19)])
        swL2 = min(lows[max(0, i - 39):max(0, i - 19)])
        if swH1 > swH2 and swL1 > swL2: bullW += 5
        elif swL1 > swL2: bullW += 2.5
        if swH1 < swH2 and swL1 < swL2: bearW += 5
        elif swH1 < swH2: bearW += 2.5
    elif not is_daily and i >= 20:
        swH1 = max(highs[max(0, i - 9):i + 1])
        swL1 = min(lows[max(0, i - 9):i + 1])
        swH2 = max(highs[max(0, i - 19):max(0, i - 9)])
        swL2 = min(lows[max(0, i - 19):max(0, i - 9)])
        if swH1 > swH2 and swL1 > swL2: bullW += 4
        elif swL1 > swL2: bullW += 2
        if swH1 < swH2 and swL1 < swL2: bearW += 4
        elif swH1 < swH2: bearW += 2

    # D12 ROC — 日足は20日ROC
    roc_period = 20 if is_daily else 10
    roc_thresh = 5 if is_daily else 3
    if i >= roc_period:
        roc = (closes[i] - closes[i - roc_period]) / closes[i - roc_period] * 100
        if roc > roc_thresh: bullW += 2
        elif roc > 0: bullW += 0.5
        if roc < -roc_thresh: bearW += 2
        elif roc < 0: bearW += 0.5

    # D13 EMA200 position (日足用追加: 長期トレンド)
    if is_daily and i >= 200:
        ema200_approx = sum(closes[i - 199:i + 1]) / 200
        if closes[i] > ema200_approx: bullW += 4
        else: bearW += 4

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

    # D19 Consecutive candles (日足用追加: 連続陽線/陰線)
    if is_daily and i >= 3:
        bullCandles = sum(1 for j in range(3) if closes[i - j] > opens[i - j])
        bearCandles = sum(1 for j in range(3) if closes[i - j] < opens[i - j])
        if bullCandles == 3: bullW += 2
        if bearCandles == 3: bearW += 2

    totalDir = bullW + bearW
    dirProb = round(max(bullW, bearW) / totalDir * 100) if totalDir > 0 else 50
    isBull = bullW > bearW
    return isBull, dirProb, bullW, bearW


# ============================================================
#  バックテスト実行（実戦仕様: SL/TP/トレーリング/確信度フィルタ）
# ============================================================
def run_backtest(data, min_score=45, is_daily=False,
                 sl_atr_mult=1.5, tp_atr_mult=3.0, trail_atr_mult=2.0,
                 max_hold=20, min_dir_prob=60, slippage=0.001, commission=0.001):
    """
    実戦仕様バックテスト:
    - ATRベースのSL/TP
    - トレーリングストップ
    - 方向確信度フィルタ (dirProb >= min_dir_prob のみエントリー)
    - 最大保有期間制限
    """
    n = len(data)
    if n < 50:
        print("[ERROR] Not enough data (need 50+ bars)")
        return None

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

    # Walk-forward: train 50%, test 50%
    split = int(n * 0.5)
    test_start = max(split, 50)

    # Trade tracking
    trades = []  # list of {entry, exit, pnl, bars, exit_reason, dir, score, dirProb}
    last_sig_bar = -10
    in_trade = False
    trade = {}

    for i in range(test_start, n):
        # --- 既存ポジション管理 ---
        if in_trade:
            bars_held = i - trade["entry_bar"]

            if trade["dir"] == "LONG":
                # トレーリングストップ更新
                new_trail = highs[i] - atr[trade["entry_bar"]] * trail_atr_mult
                trade["trail_stop"] = max(trade["trail_stop"], new_trail)
                effective_sl = max(trade["sl"], trade["trail_stop"])

                # SLヒット
                if lows[i] <= effective_sl:
                    exit_price = effective_sl * (1 - slippage)
                    pnl = (exit_price - trade["entry_price"]) / trade["entry_price"] - commission * 2
                    trades.append({**trade, "exit_price": exit_price, "pnl": pnl,
                                   "bars": bars_held, "exit_reason": "SL"})
                    in_trade = False
                    continue
                # TPヒット
                elif highs[i] >= trade["tp"]:
                    exit_price = trade["tp"] * (1 - slippage)
                    pnl = (exit_price - trade["entry_price"]) / trade["entry_price"] - commission * 2
                    trades.append({**trade, "exit_price": exit_price, "pnl": pnl,
                                   "bars": bars_held, "exit_reason": "TP"})
                    in_trade = False
                    continue
                # 最大保有期間
                elif bars_held >= max_hold:
                    exit_price = closes[i] * (1 - slippage)
                    pnl = (exit_price - trade["entry_price"]) / trade["entry_price"] - commission * 2
                    trades.append({**trade, "exit_price": exit_price, "pnl": pnl,
                                   "bars": bars_held, "exit_reason": "TIME"})
                    in_trade = False
                    continue

            else:  # SHORT
                new_trail = lows[i] + atr[trade["entry_bar"]] * trail_atr_mult
                trade["trail_stop"] = min(trade["trail_stop"], new_trail)
                effective_sl = min(trade["sl"], trade["trail_stop"])

                if highs[i] >= effective_sl:
                    exit_price = effective_sl * (1 + slippage)
                    pnl = (trade["entry_price"] - exit_price) / trade["entry_price"] - commission * 2
                    trades.append({**trade, "exit_price": exit_price, "pnl": pnl,
                                   "bars": bars_held, "exit_reason": "SL"})
                    in_trade = False
                    continue
                elif lows[i] <= trade["tp"]:
                    exit_price = trade["tp"] * (1 + slippage)
                    pnl = (trade["entry_price"] - exit_price) / trade["entry_price"] - commission * 2
                    trades.append({**trade, "exit_price": exit_price, "pnl": pnl,
                                   "bars": bars_held, "exit_reason": "TP"})
                    in_trade = False
                    continue
                elif bars_held >= max_hold:
                    exit_price = closes[i] * (1 + slippage)
                    pnl = (trade["entry_price"] - exit_price) / trade["entry_price"] - commission * 2
                    trades.append({**trade, "exit_price": exit_price, "pnl": pnl,
                                   "bars": bars_held, "exit_reason": "TIME"})
                    in_trade = False
                    continue

        # --- 新規シグナル検出 ---
        if in_trade:
            continue
        if i >= n - 3:
            continue
        if (i - last_sig_bar) < 3:
            continue

        score, kcSqz, bbSqz, atrComp = calc_squeeze_score(
            i, closes, highs, lows, opens, vols,
            bbU, bbB, bbL, kcU, kcM, kcL, atr, rsi, ema9, ema21, volSMA)

        if score < min_score:
            continue

        isBull, dirProb, bullW, bearW = calc_direction(
            i, closes, highs, lows, opens, vols,
            rsi, ema9, ema21, atr, bbU, bbL, volSMA, is_daily=is_daily)

        # 確信度フィルタ
        if dirProb < min_dir_prob:
            continue

        # トレンドフィルタ: EMA21の傾きと逆方向はスキップ
        if i >= 5:
            ema_slope = ema21[i] - ema21[i - 5]
            if isBull and ema_slope < -closes[i] * 0.001:  # 下降トレンドでLongは禁止
                continue
            if not isBull and ema_slope > closes[i] * 0.001:  # 上昇トレンドでShortは禁止
                continue

        # 日足Long onlyフィルタ (Shortが一貫して負けるため)
        if is_daily and not isBull:
            continue

        # 次足確認: シグナル足の方向と次足の始値→終値が一致しなければスキップ
        if i + 1 < n:
            next_bull = closes[i + 1] > opens[i + 1]
            if isBull and not next_bull:
                continue
            if not isBull and next_bull:
                continue

        last_sig_bar = i
        entry_bar = i + 1  # 次足の終値でエントリー（確認後）
        if entry_bar >= n:
            continue
        entry_atr = atr[i] if atr[i] > 0 else closes[i] * 0.01

        # SL: ATRベース + スイングLow/High の遠い方を採用（浅すぎ防止）
        swing_lookback = 10
        if isBull:
            entry_price = closes[entry_bar] * (1 + slippage)
            atr_sl = entry_price - entry_atr * sl_atr_mult
            swing_sl = min(lows[max(0, i - swing_lookback):i + 1])  # 直近安値
            sl = min(atr_sl, swing_sl)  # より遠い（深い）方を採用
            tp = entry_price + entry_atr * tp_atr_mult
            trade = {
                "dir": "LONG", "entry_bar": entry_bar, "entry_price": entry_price,
                "sl": sl, "tp": tp, "trail_stop": sl,
                "score": score, "dirProb": dirProb
            }
        else:
            entry_price = closes[entry_bar] * (1 - slippage)
            atr_sl = entry_price + entry_atr * sl_atr_mult
            swing_sl = max(highs[max(0, i - swing_lookback):i + 1])  # 直近高値
            sl = max(atr_sl, swing_sl)  # より遠い（深い）方を採用
            tp = entry_price - entry_atr * tp_atr_mult
            trade = {
                "dir": "SHORT", "entry_bar": entry_bar, "entry_price": entry_price,
                "sl": sl, "tp": tp, "trail_stop": sl,
                "score": score, "dirProb": dirProb
            }
        in_trade = True

    # ============================================================
    #  結果出力
    # ============================================================
    print("=" * 60)
    print("  Big Move Predictor 実戦バックテスト")
    print("=" * 60)
    print(f"  データ:     {n} bars ({'1D' if is_daily else '4H'})")
    print(f"  テスト:     bar {test_start} ~ {n}")
    print(f"  最小スコア: {min_score}%  最小確信度: {min_dir_prob}%")
    print(f"  SL: {sl_atr_mult}×ATR  TP: {tp_atr_mult}×ATR  Trail: {trail_atr_mult}×ATR")
    print(f"  最大保有:   {max_hold}本  Slip: {slippage*100:.1f}%  手数料: {commission*100:.1f}%")

    if not trades:
        print(f"\n  シグナルなし")
        return None

    # Exit reason breakdown
    tp_count = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl_count = sum(1 for t in trades if t["exit_reason"] == "SL")
    time_count = sum(1 for t in trades if t["exit_reason"] == "TIME")

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_return = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Equity curve & drawdown
    equity = [1.0]
    for p in pnls:
        equity.append(equity[-1] * (1 + p))
    peak = equity[0]
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)

    # Sharpe
    if len(pnls) > 1:
        mean_r = sum(pnls) / len(pnls)
        std_r = (sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
        sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
    else:
        sharpe = 0

    # Calmar ratio
    calmar = (total_return / len(pnls) * 252) / max_dd if max_dd > 0 else 0

    # Consecutive losses
    max_consec_loss = 0
    cur_consec = 0
    for p in pnls:
        if p <= 0:
            cur_consec += 1
            max_consec_loss = max(max_consec_loss, cur_consec)
        else:
            cur_consec = 0

    # Avg hold time
    avg_bars = sum(t["bars"] for t in trades) / len(trades)

    print(f"\n--- トレード統計 ---")
    print(f"  トレード数:     {len(trades)}")
    print(f"  勝率:           {win_rate:.1f}%")
    print(f"  平均利益:       +{avg_win * 100:.2f}%")
    print(f"  平均損失:       {avg_loss * 100:.2f}%")
    print(f"  平均RR比:       {avg_rr:.2f}")
    print(f"  Profit Factor:  {profit_factor:.2f}")
    print(f"  累積リターン:   {total_return * 100:.2f}%")
    print(f"  最大DD:         {max_dd * 100:.2f}%")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Calmar Ratio:   {calmar:.2f}")
    print(f"  最大連敗:       {max_consec_loss}")
    print(f"  平均保有:       {avg_bars:.1f} bars")
    print(f"  最終エクイティ: {equity[-1]:.4f}")

    print(f"\n--- 決済内訳 ---")
    print(f"  TP利確:   {tp_count} ({tp_count/len(trades)*100:.0f}%)")
    print(f"  SL損切:   {sl_count} ({sl_count/len(trades)*100:.0f}%)")
    print(f"  時間切れ: {time_count} ({time_count/len(trades)*100:.0f}%)")

    # Long/Short breakdown
    longs = [t for t in trades if t["dir"] == "LONG"]
    shorts = [t for t in trades if t["dir"] == "SHORT"]
    l_wr = sum(1 for t in longs if t["pnl"] > 0) / len(longs) * 100 if longs else 0
    s_wr = sum(1 for t in shorts if t["pnl"] > 0) / len(shorts) * 100 if shorts else 0
    print(f"\n--- Long/Short ---")
    print(f"  Long:  {len(longs)}回  勝率{l_wr:.0f}%  PnL {sum(t['pnl'] for t in longs)*100:.2f}%")
    print(f"  Short: {len(shorts)}回  勝率{s_wr:.0f}%  PnL {sum(t['pnl'] for t in shorts)*100:.2f}%")

    # === 厳格判定（減点方式） ===
    print(f"\n{'=' * 60}")
    print(f"  実戦判定（厳格基準）")
    print(f"{'=' * 60}")

    # スコア100点からの減点方式（厳格）
    points = 100
    reasons = []

    # 基準1: トレード数 (最低10、理想30+)
    if len(trades) < 10:
        points -= 40
        reasons.append(f"トレード数<10({len(trades)}) 統計的に無意味")
    elif len(trades) < 20:
        points -= 15
        reasons.append(f"トレード数<20({len(trades)}) 信頼性低")
    elif len(trades) < 30:
        points -= 5
        reasons.append(f"トレード数<30({len(trades)})")

    # 基準2: PF (致命的: <1.0 = マイナス期待値)
    if profit_factor < 1.0:
        points -= 50
        reasons.append(f"PF<1.0({profit_factor:.2f}) マイナス期待値")
    elif profit_factor < 1.2:
        points -= 20
        reasons.append(f"PF<1.2({profit_factor:.2f}) 手数料負け")
    elif profit_factor < 1.5:
        points -= 10
        reasons.append(f"PF<1.5({profit_factor:.2f})")

    # 基準3: 勝率
    if win_rate < 35:
        points -= 30
        reasons.append(f"勝率<35%({win_rate:.0f}%)")
    elif win_rate < 45:
        points -= 15
        reasons.append(f"勝率<45%({win_rate:.0f}%)")

    # 基準4: MaxDD (致命的: >25%)
    if max_dd > 0.25:
        points -= 35
        reasons.append(f"最大DD>25%({max_dd*100:.0f}%) 口座壊滅リスク")
    elif max_dd > 0.15:
        points -= 15
        reasons.append(f"最大DD>15%({max_dd*100:.0f}%)")
    elif max_dd > 0.10:
        points -= 5

    # 基準5: Sharpe
    if sharpe < 0:
        points -= 25
        reasons.append(f"Sharpe<0({sharpe:.2f}) リスク対比マイナス")
    elif sharpe < 0.5:
        points -= 15
        reasons.append(f"Sharpe<0.5({sharpe:.2f})")
    elif sharpe < 1.0:
        points -= 5

    # 基準6: RR比
    if avg_rr < 0.8:
        points -= 15
        reasons.append(f"RR<0.8({avg_rr:.2f}) 損大利小")
    elif avg_rr < 1.0:
        points -= 8
        reasons.append(f"RR<1.0({avg_rr:.2f})")

    # 基準7: 最大連敗
    if max_consec_loss >= 8:
        points -= 15
        reasons.append(f"連敗{max_consec_loss}回 メンタル崩壊リスク")
    elif max_consec_loss >= 6:
        points -= 8
        reasons.append(f"連敗{max_consec_loss}回")

    # 基準8: 累積リターンがマイナス = 即D
    if total_return < 0:
        points -= 20
        reasons.append(f"マイナスリターン({total_return*100:.1f}%)")

    # ポイントからグレード変換
    if points >= 80:
        grade = "S"
    elif points >= 65:
        grade = "A"
    elif points >= 50:
        grade = "B"
    elif points >= 30:
        grade = "C"
    else:
        grade = "D"

    grade_map = {
        "S": ("S  実戦投入可", "リアル資金で運用可能。ポジションサイズ管理のみ注意"),
        "A": ("A  実戦準備完了", "小ロットで3ヶ月フォワードテスト後に本格運用"),
        "B": ("B  条件付き使用", "S級シグナルのみ、ロット1/3、SL厳守で運用可"),
        "C": ("C  要改善", "パラメータ調整必要。デモトレードのみ"),
        "D": ("D  使用不可", "マイナス期待値。実資金投入禁止"),
    }

    label, advice = grade_map.get(grade, ("?", ""))
    print(f"  グレード: {label}")
    if reasons:
        for r in reasons:
            print(f"    - {r}")
    print(f"  推奨: {advice}")

    return {
        "trades": len(trades), "win_rate": win_rate, "pf": profit_factor,
        "sharpe": sharpe, "max_dd": max_dd, "return": total_return,
        "rr": avg_rr, "grade": grade
    }


# ============================================================
#  メイン
# ============================================================
if __name__ == "__main__":
    results = []

    # 4H-C推奨設定（固定）
    CFG = {"ms": 30, "mdp": 55, "sl": 2.5, "tp": 3.0, "tr": 2.0, "mh": 15}

    # === BTC 4H (8000 1H bars → 2000 4H bars) ===
    print("#" * 60)
    print("  大量データ検証: 4H-C設定は本当に勝てるか？")
    print("#" * 60)

    print("\n[1] BTC 4H...")
    data = fetch_ohlcv("BTC_USDT", "4h")
    if data and len(data) >= 100:
        print(f"  → {len(data)} bars")
        r = run_backtest(data, min_score=CFG["ms"], is_daily=False,
                         sl_atr_mult=CFG["sl"], tp_atr_mult=CFG["tp"],
                         trail_atr_mult=CFG["tr"], max_hold=CFG["mh"], min_dir_prob=CFG["mdp"])
        if r: results.append(("BTC 4H", r))

    # === BTC 1H (8000 bars直接) ===
    print(f"\n[2] BTC 1H...")
    data_1h = fetch_ohlcv("BTC_USDT", "1h")
    if data_1h and len(data_1h) >= 100:
        print(f"  → {len(data_1h)} bars")
        r = run_backtest(data_1h, min_score=CFG["ms"], is_daily=False,
                         sl_atr_mult=CFG["sl"], tp_atr_mult=CFG["tp"],
                         trail_atr_mult=CFG["tr"], max_hold=CFG["mh"]*4, min_dir_prob=CFG["mdp"])
        if r: results.append(("BTC 1H", r))

    # === ETH 4H ===
    print(f"\n[3] ETH 4H...")
    data_eth = fetch_ohlcv("ETH_USDT", "4h")
    if data_eth and len(data_eth) >= 100:
        print(f"  → {len(data_eth)} bars")
        r = run_backtest(data_eth, min_score=CFG["ms"], is_daily=False,
                         sl_atr_mult=CFG["sl"], tp_atr_mult=CFG["tp"],
                         trail_atr_mult=CFG["tr"], max_hold=CFG["mh"], min_dir_prob=CFG["mdp"])
        if r: results.append(("ETH 4H", r))

    # === SOL 4H ===
    print(f"\n[4] SOL 4H...")
    data_sol = fetch_ohlcv("SOL_USDT", "4h")
    if data_sol and len(data_sol) >= 100:
        print(f"  → {len(data_sol)} bars")
        r = run_backtest(data_sol, min_score=CFG["ms"], is_daily=False,
                         sl_atr_mult=CFG["sl"], tp_atr_mult=CFG["tp"],
                         trail_atr_mult=CFG["tr"], max_hold=CFG["mh"], min_dir_prob=CFG["mdp"])
        if r: results.append(("SOL 4H", r))

    # === BTC 1D ===
    print(f"\n[5] BTC 1D...")
    data_1d = fetch_ohlcv("BTC_USDT", "1D")
    if data_1d and len(data_1d) >= 100:
        print(f"  → {len(data_1d)} bars")
        r = run_backtest(data_1d, min_score=CFG["ms"], is_daily=True,
                         sl_atr_mult=3.0, tp_atr_mult=99.0,
                         trail_atr_mult=1.5, max_hold=20, min_dir_prob=52)
        if r: results.append(("BTC 1D", r))

    # === 総合サマリー ===
    if results:
        total_trades = sum(r["trades"] for _, r in results)
        total_wins = sum(r["trades"] * r["win_rate"] / 100 for _, r in results)

        print(f"\n\n{'#'*60}")
        print(f"  総合サマリー（4H-C設定 全市場）")
        print("#" * 60)
        print(f"  {'市場':<12} {'G':>2} {'N':>4} {'勝率':>6} {'PF':>6} {'Sharpe':>7} {'DD':>6} {'RR':>5} {'リターン':>8}")
        print(f"  {'-'*60}")
        for label, r in results:
            print(f"  {label:<12} {r['grade']:>2} {r['trades']:>4} "
                  f"{r['win_rate']:>5.1f}% {r['pf']:>5.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_dd']*100:>5.1f}% {r['rr']:>4.2f} {r['return']*100:>+7.2f}%")
        print(f"  {'-'*60}")
        agg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        print(f"  {'合計':<12}    {total_trades:>4} {agg_wr:>5.1f}%")
        print(f"\n  結論: {'データ量で優位性確認' if total_trades >= 50 and agg_wr >= 50 else '要追加検証'}")
