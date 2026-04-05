#!/usr/bin/env python3
"""
Strategy: Statistical Mean Reversion with Z-Score

=== MARKET MICROSTRUCTURE REASONING ===

Why this should have an edge:

1. Crypto markets are dominated by retail participants who exhibit herding behavior.
   This creates OVERSHOOTS beyond fair value. Extreme z-scores (|z| > 2) indicate
   prices are 2+ standard deviations from the rolling mean -- statistically rare
   events that tend to revert.

2. Unlike traditional mean reversion (RSI < 30, Bollinger Band touch), z-score
   reversion is PURELY STATISTICAL. It doesn't rely on momentum indicators that
   are lagging by construction. It measures the raw deviation from a distributional
   norm.

3. The key difference from failed approaches:
   - RSI extreme (< 25 / > 75) gave too few signals because RSI is bounded and
     sluggish. Z-score is unbounded and responds faster to real dislocations.
   - BB/KC squeeze traded breakouts (momentum). We trade REVERSIONS (contrarian).
   - ATR-based SL/TP was the wrong exit mechanism. Mean reversion should exit
     when price RETURNS TO MEAN (z -> 0), not at a fixed distance.

4. The z-score naturally adapts to volatility regimes because the standard
   deviation in the denominator expands/contracts with the market.

=== EXIT MECHANISM (CRITICAL) ===

This is where previous strategies failed. We use z-score-based exits:
- Entry at z > 2 (SHORT) or z < -2 (LONG)
- TP when z crosses back through 0 (price returned to mean)
- SL when z exceeds 3.5 (the dislocation is getting worse, we were wrong)
- Time exit after 20 bars (reversion isn't happening, cut losses)

The SL is also z-score based, NOT ATR-based. This is important because:
- ATR SL in a mean reversion trade often gets hit by the very volatility
  that created the entry signal
- Z-score SL says "if we're now at z=3.5, the distribution has shifted
  and this isn't a temporary overshoot -- it's a new regime"

=== FAILURE MODES ===

1. TRENDING MARKETS: Price deviates and KEEPS deviating. The z-score stays
   extreme for 50+ bars in a strong trend. This is the #1 killer of mean
   reversion strategies.
   Mitigation:
   - Use a MEDIUM rolling window (50 bars, not 200) so the mean adapts
   - Add a trend filter: skip signals when EMA21 slope is steep
   - Hard time exit at 20 bars

2. FAT TAILS: Crypto returns have excess kurtosis. A z-score of 2 in a
   normal distribution is the 97.7th percentile, but in crypto it might
   only be the 90th percentile.
   Mitigation: use z > 2.0 as MINIMUM, with higher confidence at z > 2.5

3. VOLATILITY REGIME CHANGE: The rolling std is backward-looking. If vol
   spikes, z-score underestimates the true deviation.
   Mitigation: reject signals where ATR has expanded > 2x over 10 bars
   (the vol regime is changing, not just the price deviating)

4. LOW FREQUENCY: Extreme z-scores are rare by definition.
   Mitigation: use z > 2.0 not z > 3.0; use rolling window of 50 not 200

=== PARAMETERS ===

- z_lookback: 50 (rolling window for mean and std calculation)
- z_entry_threshold: 2.0 (minimum |z| for entry)
- z_exit_threshold: 0.3 (exit when |z| falls below this -- near mean)
- z_stop_threshold: 3.5 (stop when |z| exceeds this -- dislocation worsening)
- trend_filter_slope: 0.002 (skip if |EMA21 5-bar pct change| > this)
- max_hold: 20 bars
- vol_stability_check: ATR[i] < 2.0 * ATR[i-10] (reject vol regime changes)
"""

import math


def calc_zscore(closes, lookback=50):
    """
    Calculate rolling z-score: (price - rolling_mean) / rolling_std
    Returns list same length as closes, with 0.0 for insufficient data.
    """
    zscores = [0.0] * len(closes)
    for i in range(lookback, len(closes)):
        window = closes[i - lookback + 1:i + 1]
        mean = sum(window) / len(window)
        var = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(var) if var > 0 else 0
        if std > 0:
            zscores[i] = (closes[i] - mean) / std
        else:
            zscores[i] = 0.0
    return zscores


def strat_zscore_reversion(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                            bbU, bbB, bbL, vols, volSMA, opens,
                            _zscore_cache={}):
    """
    Z-Score Mean Reversion Strategy.

    Signal function matching the unified backtest interface.

    IMPORTANT: This strategy's exits are z-score based. The standard
    run_strategy() engine uses ATR-based SL/TP which is suboptimal.
    For best results, use run_strategy_zscore() below.
    For compatibility with the standard engine, we still return signals
    and rely on conservative ATR params (wide SL, moderate TP).
    """
    if i < 70:  # need lookback + buffer
        return None

    # --- Compute z-scores (cached) ---
    cache_key = len(closes)
    if cache_key not in _zscore_cache:
        _zscore_cache.clear()
        _zscore_cache[cache_key] = calc_zscore(closes, lookback=50)
    zscores = _zscore_cache[cache_key]

    z = zscores[i]
    a = atr[i] if atr[i] > 0 else closes[i] * 0.01

    # === FILTER 1: Reject if z-score not extreme enough ===
    if abs(z) < 2.0:
        return None

    # === FILTER 2: Trend filter -- reject if EMA21 is trending hard ===
    # A steep EMA21 means the "mean" is shifting fast and reversion is unlikely
    if i >= 5:
        ema_slope = (ema21[i] - ema21[i - 5]) / ema21[i - 5] if ema21[i - 5] > 0 else 0
        if abs(ema_slope) > 0.025:  # > 2.5% move in EMA21 over 5 bars = strong trend
            return None

    # === FILTER 3: Vol stability -- reject if ATR has doubled recently ===
    if i >= 10 and atr[i - 10] > 0:
        atr_ratio = atr[i] / atr[i - 10]
        if atr_ratio > 2.0:
            return None  # Vol regime is changing, z-score may be unreliable

    # === FILTER 4: Reversal confirmation ===
    # Don't catch a falling knife. Wait for the first sign of reversion:
    # - For LONG (z < -2): current bar closes above its open (buyers stepping in)
    #   AND z is less extreme than previous bar's z (momentum fading)
    # - For SHORT (z > 2): current bar closes below its open (sellers stepping in)
    #   AND z is less extreme than previous bar's z
    if z < -2.0:
        # Potential LONG
        bullish_bar = closes[i] > opens[i]
        z_fading = abs(z) < abs(zscores[i - 1])  # z moving toward 0
        if not (bullish_bar and z_fading):
            return None
        direction = "LONG"
    elif z > 2.0:
        # Potential SHORT
        bearish_bar = closes[i] < opens[i]
        z_fading = abs(z) < abs(zscores[i - 1])  # z moving toward 0
        if not (bearish_bar and z_fading):
            return None
        direction = "SHORT"
    else:
        return None

    # === FILTER 5: Don't enter if the dislocation is TOO extreme ===
    # z > 3.5 means the market has fundamentally shifted, not just overshot
    if abs(z) > 3.5:
        return None

    # === Confidence scoring ===
    conf = 60

    # More extreme z (but not TOO extreme) = higher confidence
    if abs(z) > 2.5:
        conf += 10
    elif abs(z) > 2.2:
        conf += 5

    # Volume spike during extreme = likely capitulation/blow-off top = better reversion odds
    if volSMA[i] > 0 and vols[i] > volSMA[i] * 1.5:
        conf += 10

    # RSI confirmation (extreme RSI + extreme z = double confirmation)
    if direction == "LONG" and rsi[i] < 35:
        conf += 5
    elif direction == "SHORT" and rsi[i] > 65:
        conf += 5

    # Wick rejection: price tested further but pulled back (rejection)
    if direction == "LONG":
        lower_wick = min(opens[i], closes[i]) - lows[i]
        body = abs(closes[i] - opens[i]) if closes[i] != opens[i] else a * 0.01
        if lower_wick > body * 1.5:
            conf += 5
    elif direction == "SHORT":
        upper_wick = highs[i] - max(opens[i], closes[i])
        body = abs(closes[i] - opens[i]) if closes[i] != opens[i] else a * 0.01
        if upper_wick > body * 1.5:
            conf += 5

    return (direction, min(90, conf))


# ============================================================
#  Custom backtest engine with z-score-based exits
# ============================================================
def run_strategy_zscore(name, data, sl_multiplier=3.5, max_hold=20,
                        slippage=0.001, commission=0.001,
                        z_lookback=50, z_entry=2.0, z_exit=0.3, z_stop=3.5,
                        cooldown=5):
    """
    Custom backtest engine for the z-score strategy.

    Key difference from the standard engine:
    - TP is NOT a fixed ATR multiple. TP triggers when z-score returns to near 0.
    - SL is NOT a fixed ATR multiple. SL triggers when z-score EXCEEDS the stop threshold.
    - We also keep an ATR-based hard stop as a circuit breaker (sl_multiplier * ATR).
    """
    from backtest import calc_rsi, calc_atr, calc_ema, calc_bb

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

    split = int(n * 0.5)
    test_start = max(split, 70)

    trades = []
    in_trade = False
    trade = {}
    last_sig = -cooldown

    for i in range(test_start, n):
        # === Position management (z-score-based) ===
        if in_trade:
            bars_held = i - trade["entry_bar"]
            z_now = zscores[i]

            if trade["dir"] == "LONG":
                # TP: z-score returned to near 0 (mean)
                if z_now >= -z_exit:
                    pnl = (closes[i] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "LONG"})
                    in_trade = False
                    continue
                # SL: z-score got MORE extreme (dislocation worsening)
                if z_now < -z_stop:
                    pnl = (closes[i] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "LONG"})
                    in_trade = False
                    continue
                # Hard ATR stop (circuit breaker)
                if lows[i] <= trade["hard_sl"]:
                    pnl = (trade["hard_sl"] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "LONG"})
                    in_trade = False
                    continue
                # Time exit
                if bars_held >= max_hold:
                    pnl = (closes[i] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TIME", "bars": bars_held, "dir": "LONG"})
                    in_trade = False
                    continue

            else:  # SHORT
                if z_now <= z_exit:
                    pnl = (trade["ep"] - closes[i] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False
                    continue
                if z_now > z_stop:
                    pnl = (trade["ep"] - closes[i] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False
                    continue
                if highs[i] >= trade["hard_sl"]:
                    pnl = (trade["ep"] - trade["hard_sl"] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False
                    continue
                if bars_held >= max_hold:
                    pnl = (trade["ep"] - closes[i] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TIME", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False
                    continue

        if in_trade or i >= n - 2 or (i - last_sig) < cooldown:
            continue

        # === Signal generation ===
        sig = strat_zscore_reversion(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                                      bbU, bbB, bbL, vols, volSMA, opens)
        if not sig:
            continue

        direction, confidence = sig
        last_sig = i
        a = atr[i] if atr[i] > 0 else closes[i] * 0.01

        if direction == "LONG":
            ep = closes[i] * (1 + slippage)
            trade = {
                "dir": "LONG",
                "entry_bar": i,
                "ep": ep,
                "entry_z": zscores[i],
                "hard_sl": ep - a * sl_multiplier,  # circuit breaker only
            }
        else:
            ep = closes[i] * (1 - slippage)
            trade = {
                "dir": "SHORT",
                "entry_bar": i,
                "ep": ep,
                "entry_z": zscores[i],
                "hard_sl": ep + a * sl_multiplier,
            }
        in_trade = True

    # === Results ===
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
    sl_n = sum(1 for t in trades if t["reason"] == "SL")
    tm_n = sum(1 for t in trades if t["reason"] == "TIME")

    # Grade
    pts = 100
    if len(trades) < 10:
        pts -= 40
    elif len(trades) < 20:
        pts -= 15
    elif len(trades) < 30:
        pts -= 5
    if pf < 1.0:
        pts -= 50
    elif pf < 1.2:
        pts -= 20
    elif pf < 1.5:
        pts -= 10
    if wr < 35:
        pts -= 30
    elif wr < 45:
        pts -= 15
    if mdd > 0.25:
        pts -= 35
    elif mdd > 0.15:
        pts -= 15
    elif mdd > 0.10:
        pts -= 5
    if sharpe < 0:
        pts -= 25
    elif sharpe < 0.5:
        pts -= 15
    elif sharpe < 1.0:
        pts -= 5
    if rr < 0.8:
        pts -= 15
    elif rr < 1.0:
        pts -= 8
    if ret < 0:
        pts -= 20

    if pts >= 80:
        grade = "S"
    elif pts >= 65:
        grade = "A"
    elif pts >= 50:
        grade = "B"
    elif pts >= 30:
        grade = "C"
    else:
        grade = "D"

    return {
        "name": name, "n": len(trades), "wr": wr, "pf": pf, "sharpe": sharpe,
        "mdd": mdd, "rr": rr, "ret": ret, "grade": grade,
        "tp": tp_n, "sl": sl_n, "tm": tm_n, "avg_w": avg_w, "avg_l": avg_l
    }


# === Parameter sets for testing with the CUSTOM engine ===
ZSCORE_PARAMS = [
    # (name, z_lookback, z_entry, z_exit, z_stop, sl_multiplier, max_hold, cooldown)
    ("ZScore-A Standard",  50, 2.0, 0.3, 3.5, 3.5, 20, 5),
    ("ZScore-B Tight",     30, 2.0, 0.5, 3.0, 3.0, 15, 3),
    ("ZScore-C Extreme",   50, 2.5, 0.3, 4.0, 4.0, 25, 5),
    ("ZScore-D FastMean",  50, 2.0, 0.0, 3.5, 3.5, 12, 3),  # exit at z=0 exactly
    ("ZScore-E Wide",      80, 2.0, 0.3, 3.5, 4.0, 30, 8),  # longer lookback
]

# === Parameter sets for testing with the STANDARD engine ===
# Use wide SL (strategy is contrarian, needs room) and moderate TP
ZSCORE_STANDARD_PARAMS = [
    # (name, sl_atr, tp_atr, trail_atr, max_hold)
    ("ZScore-Std-A",  3.0, 2.0, 2.0, 20),
    ("ZScore-Std-B",  2.5, 1.5, 1.5, 15),
    ("ZScore-Std-C",  3.5, 3.0, 2.5, 25),
]
