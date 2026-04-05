#!/usr/bin/env python3
"""
Strategy: Volatility Regime Transition

=== MARKET MICROSTRUCTURE REASONING ===

Why this should have an edge:

Crypto markets exhibit strong volatility clustering (GARCH effects). Low-volatility
regimes are followed by high-volatility regimes, and the TRANSITION is predictable
in a statistical sense even though the DIRECTION is not. The key insight:

1. Low-vol compression happens when market makers narrow spreads and liquidity pools
   on both sides of the book. This creates a "spring" effect.

2. When vol expands from a compressed state, the first directional move tends to
   PERSIST because:
   - Stop-losses clustered just outside the compression range get triggered
   - Market makers widen spreads, reducing liquidity and amplifying the move
   - Momentum traders pile in on the breakout

3. We DON'T predict direction during compression. We wait for the vol expansion
   to START, then ride the direction it chooses.

4. The asymmetry: we enter AFTER direction is revealed but BEFORE the cascade
   of stops and momentum has fully played out.

=== FAILURE MODES ===

1. CHOPPY EXPANSION: Vol expands but no clear direction -- price whipsaws.
   Mitigation: require directional commitment (3+ bars in same direction).

2. LATE ENTRY: By the time we confirm direction, the move is mostly done.
   Mitigation: use fast vol measures (5-bar realized vol) and enter on the
   first bar that shows both vol expansion AND directional commitment.

3. FALSE COMPRESSION: Some "low vol" periods are just consolidation before
   a continuation that we miss because we wait too long.
   Mitigation: track the vol percentile rank, not absolute level.

4. EXTENDED LOW VOL: Market stays compressed for a long time. We keep
   entering on false "expansion" signals.
   Mitigation: require vol ratio > threshold (not just "higher than before").

=== PARAMETERS ===

- vol_lookback: 20 (rolling window for realized vol calculation)
- vol_rank_lookback: 100 (window for percentile rank of current vol)
- compression_percentile: 20 (vol must be below 20th percentile to be "compressed")
- expansion_ratio: 1.8 (current vol must be 1.8x the compression vol to trigger)
- direction_bars: 2 (number of consecutive bars in same direction to confirm)
- SL: 1.5 ATR (tight -- we should be right quickly if the thesis is correct)
- TP: dynamically set at 2.5-4.0 ATR based on how extreme the compression was
  (tighter compression = bigger expected move = wider TP)
- max_hold: 15 bars (vol expansion moves play out in 2-3 days on 4H)
- trail: 1.0 ATR (aggressive trailing to lock in the burst)
"""

import math


def calc_realized_vol(closes, period=20):
    """
    Realized volatility = annualized std of log returns.
    Returns a list the same length as closes.
    For indices < period, returns 0.
    """
    import math
    rvol = [0.0] * len(closes)
    for i in range(period, len(closes)):
        log_rets = []
        for j in range(i - period + 1, i + 1):
            if closes[j - 1] > 0 and closes[j] > 0:
                log_rets.append(math.log(closes[j] / closes[j - 1]))
        if len(log_rets) < 2:
            continue
        mean_r = sum(log_rets) / len(log_rets)
        var = sum((r - mean_r) ** 2 for r in log_rets) / (len(log_rets) - 1)
        rvol[i] = math.sqrt(var)
    return rvol


def calc_vol_percentile_rank(rvol, i, lookback=100):
    """
    What percentile is the current realized vol relative to the last `lookback` values?
    Returns 0-100.
    """
    start = max(0, i - lookback + 1)
    window = [v for v in rvol[start:i + 1] if v > 0]
    if len(window) < 10:
        return 50  # not enough data, return neutral
    current = rvol[i]
    rank = sum(1 for v in window if v <= current)
    return (rank / len(window)) * 100


def strat_vol_regime(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                     bbU, bbB, bbL, vols, volSMA, opens,
                     # Pre-computed arrays passed via closure or computed inline:
                     _rvol_cache={}):
    """
    Volatility Regime Transition Strategy.

    Signal function matching the unified backtest interface.
    """
    if i < 120:  # need enough history for vol percentile
        return None

    # --- Compute realized vol (cached) ---
    # We compute the full array once and cache it keyed by the data length
    cache_key = len(closes)
    if cache_key not in _rvol_cache:
        _rvol_cache.clear()
        _rvol_cache[cache_key] = calc_realized_vol(closes, period=20)
    rvol = _rvol_cache[cache_key]

    if rvol[i] <= 0 or rvol[i - 1] <= 0:
        return None

    a = atr[i] if atr[i] > 0 else closes[i] * 0.01

    # === STEP 1: Was there a recent compression? ===
    # Look back 3-20 bars for a period where vol was below 25th percentile
    compression_found = False
    compression_vol = 0
    compression_bar = 0
    for j in range(i - 20, i - 1):
        if j < 100:
            continue
        pct = calc_vol_percentile_rank(rvol, j, lookback=100)
        if pct < 25:  # Below 25th percentile = compressed
            compression_found = True
            compression_vol = rvol[j]
            compression_bar = j
            break  # take the earliest compression in the window

    if not compression_found or compression_vol <= 0:
        return None

    # === STEP 2: Is vol NOW expanding? ===
    vol_ratio = rvol[i] / compression_vol
    if vol_ratio < 1.5:
        return None  # Not enough expansion yet

    # Check that vol is INCREASING over the last 2 bars (relaxed from 3)
    vol_increasing = (rvol[i] > rvol[i - 1])
    if not vol_increasing:
        return None

    # === STEP 3: Determine direction via price action ===
    # We need directional commitment: 2+ bars closing in the same direction
    # with increasing range (the move is accelerating, not fading)

    bull_count = 0
    bear_count = 0
    for j in range(i - 2, i + 1):  # last 3 bars including current
        if closes[j] > opens[j]:
            bull_count += 1
        elif closes[j] < opens[j]:
            bear_count += 1

    # Also check net move over the last 3 bars
    net_move = closes[i] - closes[i - 3]
    net_move_pct = net_move / closes[i - 3] if closes[i - 3] > 0 else 0

    direction = None
    if bull_count >= 2 and net_move_pct > 0.003:  # 2+ bull bars AND net up > 0.3%
        direction = "LONG"
    elif bear_count >= 2 and net_move_pct < -0.003:  # 2+ bear bars AND net down > 0.3%
        direction = "SHORT"
    else:
        return None  # No clear direction yet

    # === STEP 4: Anti-exhaustion filter ===
    # If the move from compression has already gone > 3 ATR, we're too late
    price_at_compression = closes[compression_bar]
    move_since_compression = abs(closes[i] - price_at_compression) / a
    if move_since_compression > 3.5:
        return None

    # === STEP 5: Confidence scoring ===
    conf = 60

    # Stronger compression (lower percentile) = higher confidence
    min_pct = min(calc_vol_percentile_rank(rvol, j, 100)
                  for j in range(compression_bar, min(compression_bar + 3, i)))
    if min_pct < 10:
        conf += 10
    elif min_pct < 15:
        conf += 5

    # Higher vol ratio = stronger expansion signal
    if vol_ratio > 2.5:
        conf += 5
    if vol_ratio > 3.0:
        conf += 5

    # Trend alignment
    if direction == "LONG" and closes[i] > ema21[i]:
        conf += 5
    elif direction == "SHORT" and closes[i] < ema21[i]:
        conf += 5

    # Volume confirmation
    if volSMA[i] > 0 and vols[i] > volSMA[i] * 1.3:
        conf += 5

    return (direction, min(90, conf))


# === Recommended SL/TP parameters for run_strategy() ===
# These should be passed to run_strategy:
#   sl_atr=1.5, tp_atr=3.0, trail_atr=1.0, max_hold=15
#
# For tighter compression (percentile < 10), consider tp_atr=4.0
# The strategy function returns confidence but the dynamic TP
# should ideally be implemented in a custom trade manager.
# For the standard engine, tp_atr=3.0 is a good compromise.

VOL_REGIME_PARAMS = [
    # (name, sl_atr, tp_atr, trail_atr, max_hold)
    ("VolRegime-A Tight",   1.2, 2.5, 0.8, 12),
    ("VolRegime-B Standard", 1.5, 3.0, 1.0, 15),
    ("VolRegime-C Wide",    2.0, 3.5, 1.2, 18),
    ("VolRegime-D Asymm",   1.0, 4.0, 1.0, 15),  # tight SL, wide TP
    ("VolRegime-E Fast",    1.5, 2.0, 0.8, 8),    # quick in-and-out
]


# ============================================================
#  Custom backtest engine for Vol Regime with momentum-based exits
# ============================================================
def run_strategy_vol_regime(name, data, sl_atr=2.0, max_hold=20,
                             slippage=0.001, commission=0.001,
                             cooldown=5):
    """
    Custom backtest engine for the vol regime strategy.

    Key difference from standard engine:
    - TP is NOT a fixed ATR multiple. Instead we exit when:
      1. Momentum fades: the realized vol starts DECREASING (the burst is over)
         AND we have a profit
      2. Price reverses by 1 ATR from the peak since entry (trailing stop)
    - SL is ATR-based but WIDER (2.0 ATR) to give the burst room
    - Time exit at 20 bars
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
    rvol = calc_realized_vol(closes, period=20)

    split = int(n * 0.5)
    test_start = max(split, 120)

    trades = []
    in_trade = False
    trade = {}
    last_sig = -cooldown

    for i in range(test_start, n):
        if in_trade:
            bars_held = i - trade["entry_bar"]
            a_entry = atr[trade["entry_bar"]] if atr[trade["entry_bar"]] > 0 else closes[trade["entry_bar"]] * 0.01

            # Check if vol is now DECREASING (burst fading)
            vol_fading = (rvol[i] < rvol[i - 1] < rvol[i - 2]) if i >= 2 else False

            if trade["dir"] == "LONG":
                # Update peak
                trade["peak"] = max(trade["peak"], highs[i])
                current_pnl = (closes[i] - trade["ep"]) / trade["ep"]

                # Trailing stop: 1.2 ATR from peak
                trail_stop = trade["peak"] - a_entry * 1.2
                trail_stop = max(trail_stop, trade["hard_sl"])

                # TP: vol fading + profitable
                if vol_fading and current_pnl > 0.005 and bars_held >= 3:
                    pnl = (closes[i] * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "LONG"})
                    in_trade = False
                    continue
                # Trailing stop
                if lows[i] <= trail_stop:
                    pnl = (trail_stop * (1 - slippage) - trade["ep"]) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL" if current_pnl < 0 else "TP",
                                   "bars": bars_held, "dir": "LONG"})
                    in_trade = False
                    continue
                # Hard SL
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
                trade["peak"] = min(trade["peak"], lows[i])
                current_pnl = (trade["ep"] - closes[i]) / trade["ep"]

                trail_stop = trade["peak"] + a_entry * 1.2
                trail_stop = min(trail_stop, trade["hard_sl"])

                if vol_fading and current_pnl > 0.005 and bars_held >= 3:
                    pnl = (trade["ep"] - closes[i] * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "TP", "bars": bars_held, "dir": "SHORT"})
                    in_trade = False
                    continue
                if highs[i] >= trail_stop:
                    pnl = (trade["ep"] - trail_stop * (1 + slippage)) / trade["ep"] - commission * 2
                    trades.append({"pnl": pnl, "reason": "SL" if current_pnl < 0 else "TP",
                                   "bars": bars_held, "dir": "SHORT"})
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

        # Signal
        sig = strat_vol_regime(i, closes, highs, lows, rsi, atr, ema9, ema21, ema50,
                                bbU, bbB, bbL, vols, volSMA, opens)
        if not sig:
            continue

        direction, confidence = sig
        last_sig = i
        a = atr[i] if atr[i] > 0 else closes[i] * 0.01

        if direction == "LONG":
            ep = closes[i] * (1 + slippage)
            trade = {
                "dir": "LONG", "entry_bar": i, "ep": ep,
                "hard_sl": ep - a * sl_atr,
                "peak": highs[i],
            }
        else:
            ep = closes[i] * (1 - slippage)
            trade = {
                "dir": "SHORT", "entry_bar": i, "ep": ep,
                "hard_sl": ep + a * sl_atr,
                "peak": lows[i],
            }
        in_trade = True

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


VOL_REGIME_CUSTOM_PARAMS = [
    # (name, sl_atr, max_hold, cooldown)
    ("VolRegime-Custom-A",  2.0, 15, 5),
    ("VolRegime-Custom-B",  2.5, 20, 5),
    ("VolRegime-Custom-C",  3.0, 20, 3),
    ("VolRegime-Custom-D",  1.5, 12, 3),
]
