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
#  戦略シグナル生成
# ============================================================

def find_swing_highs(highs, order=5):
    """スイングハイを検出"""
    swings = []
    for i in range(order, len(highs) - order):
        if all(highs[i] >= highs[i-j] for j in range(1, order+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, order+1)):
            swings.append((i, highs[i]))
    return swings

def find_swing_lows(lows, order=5):
    """スイングローを検出"""
    swings = []
    for i in range(order, len(lows) - order):
        if all(lows[i] <= lows[i-j] for j in range(1, order+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, order+1)):
            swings.append((i, lows[i]))
    return swings


def strat_rsi_divergence(i, closes, highs, lows, rsi, atr, ema50):
    """
    戦略1: RSIダイバージェンス
    - Bullish: 価格が安値更新 + RSIが安値切り上げ → LONG
    - Bearish: 価格が高値更新 + RSIが高値切り下げ → SHORT
    - フィルタ: RSI極端域(30以下/70以上)でのみ有効
    """
    if i < 30:
        return None
    lookback = 20

    # Bullish divergence
    # 直近の安値2つを比較
    recent_lows = []
    for j in range(i - lookback, i - 2):
        if j < 2:
            continue
        if lows[j] < lows[j-1] and lows[j] < lows[j-2] and lows[j] < lows[j+1] and lows[j] < lows[j+2]:
            recent_lows.append((j, lows[j], rsi[j]))
    
    if len(recent_lows) >= 1:
        prev = recent_lows[-1]
        # 現在足が安値圏で、前の安値より価格が低いがRSIは高い
        cur_low_zone = min(lows[i-2:i+1])
        cur_rsi = min(rsi[i-2:i+1])
        if (cur_low_zone < prev[1] and  # 価格は安値更新
            cur_rsi > prev[2] and        # RSIは切り上げ
            rsi[i] < 35 and              # RSI極端域
            closes[i] > closes[i-1]):    # 反発の兆し
            return ("LONG", 70 + min(20, (35 - rsi[i]) * 2))

    # Bearish divergence
    recent_highs = []
    for j in range(i - lookback, i - 2):
        if j < 2:
            continue
        if highs[j] > highs[j-1] and highs[j] > highs[j-2] and highs[j] > highs[j+1] and highs[j] > highs[j+2]:
            recent_highs.append((j, highs[j], rsi[j]))
    
    if len(recent_highs) >= 1:
        prev = recent_highs[-1]
        cur_high_zone = max(highs[i-2:i+1])
        cur_rsi = max(rsi[i-2:i+1])
        if (cur_high_zone > prev[1] and
            cur_rsi < prev[2] and
            rsi[i] > 65 and
            closes[i] < closes[i-1]):
            return ("SHORT", 70 + min(20, (rsi[i] - 65) * 2))

    return None


def strat_ema_cross(i, closes, ema9, ema21, ema50, atr, rsi):
    """
    戦略2: EMAクロス + トレンドフィルタ
    - EMA9がEMA21をゴールデンクロス + 価格>EMA50 → LONG
    - EMA9がEMA21をデッドクロス + 価格<EMA50 → SHORT
    - RSI40-60の中立域では見送り
    """
    if i < 2 or i < 50:
        return None
    
    # ゴールデンクロス
    if ema9[i-1] <= ema21[i-1] and ema9[i] > ema21[i]:
        if closes[i] > ema50[i] and rsi[i] > 50:
            conf = 60 + min(20, (rsi[i] - 50))
            return ("LONG", conf)
    
    # デッドクロス
    if ema9[i-1] >= ema21[i-1] and ema9[i] < ema21[i]:
        if closes[i] < ema50[i] and rsi[i] < 50:
            conf = 60 + min(20, (50 - rsi[i]))
            return ("SHORT", conf)
    
    return None


def strat_rsi_extreme(i, closes, highs, lows, rsi, atr, bbL, bbU, vol, volSMA):
    """
    戦略3: RSI極端値 + ボリューム確認
    - RSI<25 + ボリュームスパイク + BB下バンド下 → LONG
    - RSI>75 + ボリュームスパイク + BB上バンド上 → SHORT
    """
    if i < 20:
        return None
    
    vol_spike = vol[i] > volSMA[i] * 1.5 if volSMA[i] > 0 else False
    
    if rsi[i] < 25 and closes[i] < bbL[i] and vol_spike:
        if closes[i] > closes[i-1]:  # 反発の兆し
            conf = 75 + min(15, (25 - rsi[i]) * 3)
            return ("LONG", conf)
    
    if rsi[i] > 75 and closes[i] > bbU[i] and vol_spike:
        if closes[i] < closes[i-1]:  # 反落の兆し
            conf = 75 + min(15, (rsi[i] - 75) * 3)
            return ("SHORT", conf)
    
    return None


def strat_trend_pullback(i, closes, highs, lows, ema21, ema50, rsi, atr):
    """
    戦略4: トレンド中の押し目/戻り（改良版）

    エントリー条件（LONG）:
    1. 強いトレンド: EMA21>EMA50 + EMA21が5本連続上昇
    2. プルバック: 安値がEMA21タッチ or EMA21-50の間
    3. RSI: 35-55（売られ過ぎではないが中立寄り）
    4. 反発確認: 陽線 + 前の足より高値更新
    5. プルバック深さ: 直近高値から1-3ATR下落
    """
    if i < 55:
        return None

    a = atr[i] if atr[i] > 0 else closes[i] * 0.01

    # === トレンド強度判定 ===
    # EMA21 > EMA50 かつ EMA21が3本連続上昇（緩和: 5→3）
    uptrend = (ema21[i] > ema50[i] and
               all(ema21[i-j] > ema21[i-j-1] for j in range(3)))
    dntrend = (ema21[i] < ema50[i] and
               all(ema21[i-j] < ema21[i-j-1] for j in range(3)))

    # トレンドの勢い（EMA21の傾き角度）
    ema_slope = abs(ema21[i] - ema21[i-5]) / a if a > 0 else 0
    if ema_slope < 0.15:  # 傾きが弱すぎる → レンジ（緩和: 0.3→0.15）
        return None

    if uptrend:
        # プルバック検出: EMA21付近まで下落
        pullback_to_ema = lows[i] <= ema21[i] + a * 0.8  # EMA21+0.8ATR以下（緩和）
        not_too_deep = lows[i] >= ema50[i] - a * 1.0     # EMA50-1ATRまで許容（緩和）

        # 直近高値からの下落幅チェック
        recent_high = max(highs[max(0,i-10):i])
        drop = recent_high - lows[i]
        good_depth = a * 0.5 < drop < a * 5.0  # 0.5-5ATRの押し（緩和）

        if pullback_to_ema and not_too_deep and good_depth:
            # 反発確認: 陽線 + 安値が前足より切り上げ
            bullish_bar = closes[i] > opens_g[i]
            higher_low = lows[i] > lows[i-1] if i > 0 else False
            rsi_ok = 25 < rsi[i] < 60  # RSI範囲拡大（緩和）

            if bullish_bar and higher_low and rsi_ok:
                # ボーナス: 下ヒゲが長い（買い圧力）
                body = abs(closes[i] - opens_g[i])
                lower_wick = min(closes[i], opens_g[i]) - lows[i]
                wick_bonus = 5 if lower_wick > body * 1.5 else 0

                conf = 60 + min(15, ema_slope * 5) + wick_bonus
                return ("LONG", min(90, conf))

    if dntrend:
        pullback_to_ema = highs[i] >= ema21[i] - a * 0.8
        not_too_deep = highs[i] <= ema50[i] + a * 1.0

        recent_low = min(lows[max(0,i-10):i])
        rise = highs[i] - recent_low
        good_depth = a * 0.5 < rise < a * 5.0

        if pullback_to_ema and not_too_deep and good_depth:
            bearish_bar = closes[i] < opens_g[i]
            lower_high = highs[i] < highs[i-1] if i > 0 else False
            rsi_ok = 40 < rsi[i] < 75  # RSI範囲拡大（緩和）

            if bearish_bar and lower_high and rsi_ok:
                body = abs(closes[i] - opens_g[i])
                upper_wick = highs[i] - max(closes[i], opens_g[i])
                wick_bonus = 5 if upper_wick > body * 1.5 else 0

                conf = 60 + min(15, ema_slope * 5) + wick_bonus
                return ("SHORT", min(90, conf))

    return None


def strat_macd_divergence(i, closes, highs, lows, ema9, ema21, rsi):
    """
    戦略5: MACDダイバージェンス
    - MACD = EMA9-EMA21 のダイバージェンス
    """
    if i < 25:
        return None
    
    macd = [ema9[j] - ema21[j] for j in range(len(ema9))]
    lookback = 15
    
    # Bullish: 価格安値更新 + MACD切り上げ
    prev_low_i = None
    for j in range(i - lookback, i - 3):
        if j < 1:
            continue
        if lows[j] < lows[j-1] and lows[j] < lows[j+1]:
            prev_low_i = j
    
    if prev_low_i and lows[i] < lows[prev_low_i] and macd[i] > macd[prev_low_i]:
        if rsi[i] < 40:
            return ("LONG", 65)
    
    # Bearish: 価格高値更新 + MACD切り下げ
    prev_high_i = None
    for j in range(i - lookback, i - 3):
        if j < 1:
            continue
        if highs[j] > highs[j-1] and highs[j] > highs[j+1]:
            prev_high_i = j
    
    if prev_high_i and highs[i] > highs[prev_high_i] and macd[i] < macd[prev_high_i]:
        if rsi[i] > 60:
            return ("SHORT", 65)
    
    return None


# ============================================================
#  戦略6: Momentum Burst v3 (Multi-Confluence Exhaustion Reversal)
# ============================================================
def strat_momentum_burst(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                         bbU, bbB, bbL, vols, volSMA, opens):
    """
    Momentum Burst v3 — Ultra-Selective Exhaustion Reversal

    Lessons from v1/v2: The market on 4H punishes almost any signal
    that fires frequently. Only EXTREME setups with multiple confluences
    have positive expectancy.

    This version requires ALL of the following simultaneously:
    1. Extreme extension: price > 2.5 ATR from 20-bar SMA
    2. RSI extreme: < 25 or > 75 (true exhaustion territory)
    3. Volume pattern: climax volume on ONE of last 3 bars (institutional
       capitulation), but current bar volume is DECLINING (exhaustion)
    4. Candlestick reversal pattern: engulfing or pin bar
    5. BB penetration: price outside Bollinger Band (statistical extreme)

    Additionally, we ONLY take the signal if the prior move was
    large enough (> 3 ATR over last 8 bars) — we need a real burst
    to have occurred, not a slow drift.

    This will fire VERY rarely — maybe 2-8 times per 1000 bars —
    but each signal carries extreme confluence.
    """
    if i < 30:
        return None

    a = atr[i] if atr[i] > 0 else closes[i] * 0.01
    if a <= 0:
        return None

    # --- Pre-filter: Was there a real burst? ---
    # Total move over last 8 bars must be > 2.5 ATR
    move_8 = closes[i] - closes[max(0, i - 8)]
    if abs(move_8) < a * 2.5:
        return None

    # --- 1. Extreme extension from mean ---
    mean_20 = sum(closes[max(0, i - 19):i + 1]) / min(20, i + 1)
    extension = (closes[i] - mean_20) / a

    # --- 2. RSI extreme ---
    rsi_val = rsi[i]

    # --- 3. Volume pattern: climax in recent bars, declining now ---
    had_climax = False
    for j in range(1, 4):
        if i - j >= 0 and volSMA[i] > 0:
            if vols[i - j] > volSMA[i] * 1.8:
                had_climax = True
                break
    vol_declining = vols[i] < vols[i - 1] if i > 0 else False

    # --- 4. Candlestick reversal pattern ---
    cur_body = closes[i] - opens[i]  # Positive = bullish
    prev_body = closes[i - 1] - opens[i - 1]
    upper_wick = highs[i] - max(closes[i], opens[i])
    lower_wick = min(closes[i], opens[i]) - lows[i]
    abs_body = abs(cur_body)

    # Engulfing: current bar body engulfs previous bar body, opposite direction
    bull_engulf = (cur_body > 0 and prev_body < 0 and
                   abs_body > abs(prev_body) * 0.9 and
                   abs_body > a * 0.3)
    bear_engulf = (cur_body < 0 and prev_body > 0 and
                   abs_body > abs(prev_body) * 0.9 and
                   abs_body > a * 0.3)

    # Pin bar / hammer: wick > 2x body
    bull_pin = lower_wick > max(abs_body, a * 0.1) * 2.0 and cur_body >= 0
    bear_pin = upper_wick > max(abs_body, a * 0.1) * 2.0 and cur_body <= 0

    # --- 5. BB penetration ---
    below_bb = closes[i] < bbL[i] or lows[i] < bbL[i]
    above_bb = closes[i] > bbU[i] or highs[i] > bbU[i]

    # === LONG (bullish reversal after bearish burst) ===
    if move_8 < 0 and extension < -2.0:
        score = 0
        if rsi_val < 28: score += 1
        if rsi_val < 20: score += 1  # Extra point for extreme
        if had_climax and vol_declining: score += 1
        if bull_engulf: score += 1
        if bull_pin: score += 1
        if below_bb: score += 1

        # Need >= 3 points from 6 possible (demanding but achievable)
        if score >= 3:
            conf = 60 + score * 5
            return ("LONG", min(90, conf))

    # === SHORT (bearish reversal after bullish burst) ===
    if move_8 > 0 and extension > 2.0:
        score = 0
        if rsi_val > 72: score += 1
        if rsi_val > 80: score += 1
        if had_climax and vol_declining: score += 1
        if bear_engulf: score += 1
        if bear_pin: score += 1
        if above_bb: score += 1

        if score >= 3:
            conf = 60 + score * 5
            return ("SHORT", min(90, conf))

    return None


# ============================================================
#  戦略7: Adaptive Channel Mean-Reversion (自己調整チャネル回帰)
# ============================================================
def _adaptive_lookback(closes, highs, lows, atr, i, min_lb=10, max_lb=40):
    """
    Compute an adaptive lookback period based on volatility regime.

    Uses volatility ratio + Kaufman efficiency ratio.
    """
    if i < max_lb + 5:
        return 20

    short_atr = sum(abs(highs[j] - lows[j]) for j in range(i - 4, i + 1)) / 5
    long_atr = sum(abs(highs[j] - lows[j]) for j in range(i - 29, i + 1)) / 30
    vol_ratio = short_atr / long_atr if long_atr > 0 else 1.0

    direction = abs(closes[i] - closes[i - 20])
    total_path = sum(abs(closes[j] - closes[j - 1]) for j in range(i - 19, i + 1))
    efficiency = direction / total_path if total_path > 0 else 0

    # High vol_ratio or high efficiency → shorter lookback
    combined = 0.5 * min(vol_ratio, 2.0) / 2.0 + 0.5 * efficiency
    lookback = int(max_lb - combined * (max_lb - min_lb))
    return max(min_lb, min(max_lb, lookback))


def strat_adaptive_channel(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                           bbU, bbB, bbL, vols, volSMA, opens):
    """
    Adaptive Channel v3 — Regime-Aware Boundary Fade

    Lessons from v1/v2: The channel concept is sound, but entries must
    be ultra-selective. V2 still fired too often (18-23 trades) with
    poor win rates.

    V3 changes:
    - ONLY trade in the "ranging" regime (low efficiency ratio).
      In trending regimes, channel boundaries get blown through.
    - Require TWO consecutive rejection bars at the boundary
      (single-bar rejections have too high failure rate)
    - Require RSI divergence at the boundary (price makes new
      channel extreme but RSI doesn't — exhaustion signal)
    - Use Bollinger Band %B as additional overextension measure

    The adaptive lookback is retained but now also determines whether
    to trade at all: if the lookback is very short (trending market),
    we skip entirely.
    """
    if i < 55:
        return None

    a = atr[i] if atr[i] > 0 else closes[i] * 0.01
    if a <= 0:
        return None

    # --- Regime detection ---
    lb = _adaptive_lookback(closes, highs, lows, atr, i)

    # If lookback is short (< 15), market is trending — skip
    if lb < 15:
        return None

    # Also compute efficiency ratio directly
    direction = abs(closes[i] - closes[i - 20])
    total_path = sum(abs(closes[j] - closes[j - 1]) for j in range(i - 19, i + 1))
    efficiency = direction / total_path if total_path > 0 else 0

    # Only trade in ranging regime (efficiency < 0.35)
    if efficiency > 0.35:
        return None

    # --- Build channel ---
    chan_high = max(highs[i - lb:i])
    chan_low = min(lows[i - lb:i])
    chan_mid = (chan_high + chan_low) / 2
    chan_width = chan_high - chan_low

    if chan_width <= 0 or chan_width < a * 1.0:
        return None

    # --- Price at boundary? ---
    at_upper = highs[i] >= chan_high - a * 0.3
    at_lower = lows[i] <= chan_low + a * 0.3

    if not at_upper and not at_lower:
        return None

    # --- Candlestick analysis ---
    upper_wick = highs[i] - max(closes[i], opens[i])
    lower_wick = min(closes[i], opens[i]) - lows[i]
    body = abs(closes[i] - opens[i])
    prev_upper_wick = highs[i-1] - max(closes[i-1], opens[i-1])
    prev_lower_wick = min(closes[i-1], opens[i-1]) - lows[i-1]
    prev_body = abs(closes[i-1] - opens[i-1])

    # --- BB %B as overextension measure ---
    bb_width = bbU[i] - bbL[i]
    bb_pctb = (closes[i] - bbL[i]) / bb_width if bb_width > 0 else 0.5

    # === SHORT at upper boundary ===
    if at_upper:
        score = 0

        # Rejection wick on current bar
        if upper_wick > max(body, a * 0.05) * 1.0:
            score += 1

        # Two-bar rejection pattern (both bars rejected at top)
        prev_at_upper = highs[i-1] >= chan_high - a * 0.5
        if prev_at_upper and prev_upper_wick > max(prev_body, a * 0.05) * 0.8:
            score += 1

        # Close back inside channel
        if closes[i] < chan_high:
            score += 1

        # Bearish close
        if closes[i] < opens[i]:
            score += 1

        # RSI overbought
        if rsi[i] > 65:
            score += 1
        if rsi[i] > 75:
            score += 1

        # RSI divergence: price at/near high but RSI lower than recent
        if i >= 10:
            recent_rsi_high = max(rsi[max(0, i-10):i])
            if rsi[i] < recent_rsi_high - 5 and highs[i] >= max(highs[max(0,i-10):i]) * 0.998:
                score += 1  # Bearish RSI divergence

        # BB %B > 0.95 (well above upper band)
        if bb_pctb > 0.90:
            score += 1

        # Low volume (false breakout indicator)
        if volSMA[i] > 0 and vols[i] < volSMA[i] * 1.0:
            score += 1

        # Need 4+ out of 9 possible
        if score >= 4:
            room = (closes[i] - chan_mid) / a
            if room > 0.3:
                conf = 55 + score * 4
                return ("SHORT", min(90, conf))

    # === LONG at lower boundary ===
    if at_lower:
        score = 0

        if lower_wick > max(body, a * 0.05) * 1.0:
            score += 1

        prev_at_lower = lows[i-1] <= chan_low + a * 0.5
        if prev_at_lower and prev_lower_wick > max(prev_body, a * 0.05) * 0.8:
            score += 1

        if closes[i] > chan_low:
            score += 1

        if closes[i] > opens[i]:
            score += 1

        if rsi[i] < 35:
            score += 1
        if rsi[i] < 25:
            score += 1

        if i >= 10:
            recent_rsi_low = min(rsi[max(0, i-10):i])
            if rsi[i] > recent_rsi_low + 5 and lows[i] <= min(lows[max(0,i-10):i]) * 1.002:
                score += 1

        if bb_pctb < 0.10:
            score += 1

        if volSMA[i] > 0 and vols[i] < volSMA[i] * 1.0:
            score += 1

        if score >= 4:
            room = (chan_mid - closes[i]) / a
            if room > 0.3:
                conf = 55 + score * 4
                return ("LONG", min(90, conf))

    return None


# ============================================================
#  統一バックテストエンジン
# ============================================================
def run_strategy(name, data, signal_func, sl_atr=1.5, tp_atr=2.0, trail_atr=1.2,
                 max_hold=12, slippage=0.001, commission=0.001):
    n = len(data)
    if n < 60:
        return None

    closes = [d["c"] for d in data]
    highs = [d["h"] for d in data]
    lows = [d["l"] for d in data]
    opens = [d["o"] for d in data]
    vols = [d["v"] for d in data]

    # Make opens available globally for strat_trend_pullback
    global opens_g
    opens_g = opens

    rsi = calc_rsi(closes)
    atr = calc_atr(highs, lows, closes)
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    ema50 = calc_ema(closes, 50)
    bbU, bbB, bbL = calc_bb(closes)
    volSMA = calc_ema(vols, 20)

    split = int(n * 0.5)
    test_start = max(split, 55)

    trades = []
    in_trade = False
    trade = {}
    last_sig = -10

    for i in range(test_start, n):
        # Position management
        if in_trade:
            bars_held = i - trade["entry_bar"]
            if trade["dir"] == "LONG":
                trade["trail_stop"] = max(trade["trail_stop"],
                                          highs[i] - atr[trade["entry_bar"]] * trail_atr)
                eff_sl = max(trade["sl"], trade["trail_stop"])
                if lows[i] <= eff_sl:
                    pnl = (eff_sl * (1-slippage) - trade["ep"]) / trade["ep"] - commission*2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "LONG"})
                    in_trade = False; continue
                if highs[i] >= trade["tp"]:
                    pnl = (trade["tp"] * (1-slippage) - trade["ep"]) / trade["ep"] - commission*2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "LONG"})
                    in_trade = False; continue
                if bars_held >= max_hold:
                    pnl = (closes[i] * (1-slippage) - trade["ep"]) / trade["ep"] - commission*2
                    trades.append({"pnl": pnl, "reason": "TIME", "bars": bars_held, "dir": "LONG"})
                    in_trade = False; continue
            else:
                trade["trail_stop"] = min(trade["trail_stop"],
                                          lows[i] + atr[trade["entry_bar"]] * trail_atr)
                eff_sl = min(trade["sl"], trade["trail_stop"])
                if highs[i] >= eff_sl:
                    pnl = (trade["ep"] - eff_sl * (1+slippage)) / trade["ep"] - commission*2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False; continue
                if lows[i] <= trade["tp"]:
                    pnl = (trade["ep"] - trade["tp"] * (1+slippage)) / trade["ep"] - commission*2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False; continue
                if bars_held >= max_hold:
                    pnl = (trade["ep"] - closes[i] * (1+slippage)) / trade["ep"] - commission*2
                    trades.append({"pnl": pnl, "reason": "TIME", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False; continue

        if in_trade or i >= n - 2 or (i - last_sig) < 5:
            continue

        # Signal
        sig = signal_func(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                          bbU, bbB, bbL, vols, volSMA, opens)
        if not sig:
            continue

        direction, confidence = sig
        last_sig = i
        a = atr[i] if atr[i] > 0 else closes[i] * 0.01

        if direction == "LONG":
            ep = closes[i] * (1 + slippage)
            trade = {"dir": "LONG", "entry_bar": i, "ep": ep,
                     "sl": ep - a * sl_atr, "tp": ep + a * tp_atr,
                     "trail_stop": ep - a * sl_atr}
        else:
            ep = closes[i] * (1 - slippage)
            trade = {"dir": "SHORT", "entry_bar": i, "ep": ep,
                     "sl": ep + a * sl_atr, "tp": ep - a * tp_atr,
                     "trail_stop": ep + a * sl_atr}
        in_trade = True

    # Results
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
        sr = (sum((p - mr)**2 for p in pnls) / (len(pnls)-1)) ** 0.5
        sharpe = (mr / sr) * (252**0.5) if sr > 0 else 0
    else:
        sharpe = 0

    tp_n = sum(1 for t in trades if t["reason"] == "TP")
    sl_n = sum(1 for t in trades if t["reason"] == "SL")
    tm_n = sum(1 for t in trades if t["reason"] == "TIME")

    # Grade (strict)
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

    return {
        "name": name, "n": len(trades), "wr": wr, "pf": pf, "sharpe": sharpe,
        "mdd": mdd, "rr": rr, "ret": ret, "grade": grade,
        "tp": tp_n, "sl": sl_n, "tm": tm_n, "avg_w": avg_w, "avg_l": avg_l
    }


# ============================================================
#  メイン
# ============================================================
if __name__ == "__main__":
    print("#" * 60)
    print("  マルチ戦略バックテスト（厳格判定）")
    print("#" * 60)

    # Wrapper functions that match the unified interface
    def sig_rsi_div(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
        return strat_rsi_divergence(i, C, H, L, rsi, atr, e50)

    def sig_ema_cross(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
        return strat_ema_cross(i, C, e9, e21, e50, atr, rsi)

    def sig_rsi_extreme(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
        return strat_rsi_extreme(i, C, H, L, rsi, atr, bbL, bbU, vol, volSMA)

    def sig_pullback(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
        return strat_trend_pullback(i, C, H, L, e21, e50, rsi, atr)

    def sig_macd_div(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
        return strat_macd_divergence(i, C, H, L, e9, e21, rsi)

    def sig_mom_burst(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
        return strat_momentum_burst(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O)

    def sig_adapt_chan(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
        return strat_adaptive_channel(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O)

    strategies = [
        # 押し目/戻り改良版のパラメータ探索
        ("PB-A タイト",     sig_pullback,   1.2, 1.5, 1.0, 8),
        ("PB-B 標準",       sig_pullback,   1.5, 2.0, 1.2, 10),
        ("PB-C 広SL",       sig_pullback,   2.0, 2.0, 1.5, 10),
        ("PB-D 広TP",       sig_pullback,   1.5, 2.5, 1.5, 12),
        ("PB-E RR重視",     sig_pullback,   1.0, 2.5, 1.2, 10),
        ("PB-F 長保有",     sig_pullback,   2.0, 3.0, 1.8, 20),
        ("PB-G 即利確",     sig_pullback,   1.5, 1.0, 0.8, 5),
        # 他の戦略も残す（比較用）
        ("RSIダイバ",       sig_rsi_div,    1.5, 2.0, 1.2, 12),
        ("EMAクロス",       sig_ema_cross,  1.5, 2.0, 1.2, 12),
        ("RSI極端値",       sig_rsi_extreme,1.5, 2.0, 1.2, 8),
        ("MACDダイバ",      sig_macd_div,   1.5, 2.0, 1.2, 12),
        # --- NEW: Momentum Burst v3 (Exhaustion Reversal) ---
        ("MB-A タイト",     sig_mom_burst,  1.0, 1.5, 0.8, 6),
        ("MB-B 標準",       sig_mom_burst,  1.5, 2.0, 1.2, 8),
        ("MB-C 広幅",       sig_mom_burst,  2.0, 3.0, 1.5, 12),
        ("MB-D RR重視",     sig_mom_burst,  1.0, 3.0, 1.0, 10),
        ("MB-E 即利確",     sig_mom_burst,  1.2, 1.0, 0.8, 4),
        ("MB-F 広SL",       sig_mom_burst,  2.5, 2.0, 2.0, 10),
        ("MB-G 超広SL",     sig_mom_burst,  3.0, 2.5, 2.5, 15),
        ("MB-H ミドル",     sig_mom_burst,  1.8, 2.5, 1.5, 8),
        # --- NEW: Adaptive Channel v3 (Regime-Aware Fade) ---
        ("AC-A タイト",     sig_adapt_chan,  1.0, 1.5, 0.8, 6),
        ("AC-B 標準",       sig_adapt_chan,  1.5, 2.0, 1.2, 8),
        ("AC-C 広幅",       sig_adapt_chan,  2.0, 3.0, 1.5, 12),
        ("AC-D RR重視",     sig_adapt_chan,  1.0, 3.0, 1.0, 10),
        ("AC-E 即利確",     sig_adapt_chan,  1.2, 1.0, 0.8, 4),
        ("AC-F 広SL",       sig_adapt_chan,  2.5, 2.0, 2.0, 10),
        ("AC-G 超広SL",     sig_adapt_chan,  3.0, 2.5, 2.5, 15),
        ("AC-H ミドル",     sig_adapt_chan,  1.8, 2.5, 1.5, 8),
    ]

    all_results = []

    for sym, sym_name in [("BTC_USDT", "BTC"), ("ETH_USDT", "ETH"), ("SOL_USDT", "SOL")]:
        print(f"\n{'='*60}")
        print(f"  {sym_name} 4H")
        print(f"{'='*60}")
        data = fetch_ohlcv(sym, "4h")
        if not data or len(data) < 100:
            print(f"  [SKIP] データ不足")
            continue
        print(f"  {len(data)} bars")

        for sname, sfunc, sl, tp, tr, mh in strategies:
            label = f"{sym_name} {sname}"
            r = run_strategy(label, data, sfunc, sl_atr=sl, tp_atr=tp,
                             trail_atr=tr, max_hold=mh)
            if r:
                all_results.append(r)

    # Summary
    print(f"\n\n{'#'*60}")
    print(f"  総合サマリー（厳格判定）")
    print("#" * 60)
    print(f"  {'戦略':<20} {'G':>2} {'N':>4} {'勝率':>6} {'PF':>6} {'Sharpe':>7} {'DD':>6} {'RR':>5} {'Ret':>8} {'TP':>3} {'SL':>3}")
    print(f"  {'-'*74}")
    for r in sorted(all_results, key=lambda x: -x["ret"]):
        print(f"  {r['name']:<20} {r['grade']:>2} {r['n']:>4} "
              f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['sharpe']:>7.2f} "
              f"{r['mdd']*100:>5.1f}% {r['rr']:>4.2f} {r['ret']*100:>+7.2f}% "
              f"{r['tp']:>3} {r['sl']:>3}")

    # Best
    profitable = [r for r in all_results if r["ret"] > 0 and r["pf"] > 1.0]
    print(f"\n  プラス期待値: {len(profitable)}/{len(all_results)} 戦略")
    if profitable:
        best = max(profitable, key=lambda x: x["sharpe"])
        print(f"  ベスト: {best['name']} (Sharpe {best['sharpe']:.2f}, PF {best['pf']:.2f}, {best['grade']})")
    else:
        print(f"  *** プラス期待値の戦略なし ***")
