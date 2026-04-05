#!/usr/bin/env python3
"""
Quantitative Strategy Backtest Runner

Tests two new strategies against BTC/ETH/SOL 4H data:
1. Volatility Regime Transition
2. Z-Score Mean Reversion (with both standard and custom engines)

Usage: python3 backtest_quant.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import fetch_ohlcv, run_strategy, calc_rsi, calc_atr, calc_ema, calc_bb
from strat_vol_regime import strat_vol_regime, VOL_REGIME_PARAMS, run_strategy_vol_regime, VOL_REGIME_CUSTOM_PARAMS
from strat_zscore_reversion import (
    strat_zscore_reversion, run_strategy_zscore,
    ZSCORE_PARAMS, ZSCORE_STANDARD_PARAMS
)


def main():
    print("#" * 60)
    print("  Quantitative Strategy Backtest")
    print("  Vol Regime Transition + Z-Score Mean Reversion")
    print("#" * 60)

    all_results = []

    for sym, sym_name in [("BTC_USDT", "BTC"), ("ETH_USDT", "ETH"), ("SOL_USDT", "SOL")]:
        print(f"\n{'=' * 60}")
        print(f"  {sym_name} 4H")
        print(f"{'=' * 60}")
        data = fetch_ohlcv(sym, "4h")
        if not data or len(data) < 100:
            print(f"  [SKIP] Insufficient data")
            continue
        print(f"  {len(data)} bars loaded")

        # --- Strategy 1: Volatility Regime Transition ---
        # Uses the standard engine with ATR-based SL/TP
        print(f"\n  --- Vol Regime Transition ---")

        # Clear the cache between symbols
        strat_vol_regime.__defaults__[-1].clear()

        def sig_vol_regime(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
            return strat_vol_regime(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O)

        for pname, sl, tp, tr, mh in VOL_REGIME_PARAMS:
            label = f"{sym_name} {pname}"
            r = run_strategy(label, data, sig_vol_regime, sl_atr=sl, tp_atr=tp,
                             trail_atr=tr, max_hold=mh)
            if r:
                all_results.append(r)
                print(f"    {pname}: {r['grade']} | {r['n']} trades | "
                      f"WR={r['wr']:.1f}% | PF={r['pf']:.2f} | "
                      f"Ret={r['ret']*100:+.2f}%")
            else:
                print(f"    {pname}: No trades")

        # --- Strategy 1b: Volatility Regime with Custom Engine ---
        print(f"\n  --- Vol Regime Transition (Custom Engine) ---")
        strat_vol_regime.__defaults__[-1].clear()

        for pname, sl, mh, cd in VOL_REGIME_CUSTOM_PARAMS:
            label = f"{sym_name} {pname}"
            r = run_strategy_vol_regime(label, data, sl_atr=sl, max_hold=mh, cooldown=cd)
            if r:
                all_results.append(r)
                print(f"    {pname}: {r['grade']} | {r['n']} trades | "
                      f"WR={r['wr']:.1f}% | PF={r['pf']:.2f} | "
                      f"Ret={r['ret']*100:+.2f}%")
            else:
                print(f"    {pname}: No trades")

        # --- Strategy 2: Z-Score Mean Reversion ---
        print(f"\n  --- Z-Score Mean Reversion (Standard Engine) ---")

        # Clear the cache between symbols
        strat_zscore_reversion.__defaults__[-1].clear()

        def sig_zscore(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O):
            return strat_zscore_reversion(i, C, H, L, rsi, atr, e9, e21, e50, bbU, bbB, bbL, vol, volSMA, O)

        for pname, sl, tp, tr, mh in ZSCORE_STANDARD_PARAMS:
            label = f"{sym_name} {pname}"
            r = run_strategy(label, data, sig_zscore, sl_atr=sl, tp_atr=tp,
                             trail_atr=tr, max_hold=mh)
            if r:
                all_results.append(r)
                print(f"    {pname}: {r['grade']} | {r['n']} trades | "
                      f"WR={r['wr']:.1f}% | PF={r['pf']:.2f} | "
                      f"Ret={r['ret']*100:+.2f}%")
            else:
                print(f"    {pname}: No trades")

        # --- Strategy 2b: Z-Score with Custom Engine (z-score based exits) ---
        print(f"\n  --- Z-Score Mean Reversion (Custom Z-Engine) ---")

        # Clear cache again
        strat_zscore_reversion.__defaults__[-1].clear()

        for pname, z_lb, z_entry, z_exit, z_stop, sl_mult, mh, cd in ZSCORE_PARAMS:
            label = f"{sym_name} {pname}"
            r = run_strategy_zscore(label, data, sl_multiplier=sl_mult, max_hold=mh,
                                     z_lookback=z_lb, z_entry=z_entry, z_exit=z_exit,
                                     z_stop=z_stop, cooldown=cd)
            if r:
                all_results.append(r)
                print(f"    {pname}: {r['grade']} | {r['n']} trades | "
                      f"WR={r['wr']:.1f}% | PF={r['pf']:.2f} | "
                      f"Ret={r['ret']*100:+.2f}%")
            else:
                print(f"    {pname}: No trades")

    # === Summary ===
    print(f"\n\n{'#' * 60}")
    print(f"  OVERALL SUMMARY")
    print("#" * 60)

    if not all_results:
        print("  No results to display.")
        return

    print(f"  {'Strategy':<28} {'G':>2} {'N':>4} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
          f"{'DD':>6} {'RR':>5} {'Ret':>8} {'TP':>3} {'SL':>3} {'TM':>3}")
    print(f"  {'-' * 90}")

    for r in sorted(all_results, key=lambda x: -x["ret"]):
        print(f"  {r['name']:<28} {r['grade']:>2} {r['n']:>4} "
              f"{r['wr']:>5.1f}% {r['pf']:>5.2f} {r['sharpe']:>7.2f} "
              f"{r['mdd']*100:>5.1f}% {r['rr']:>4.2f} {r['ret']*100:>+7.2f}% "
              f"{r['tp']:>3} {r['sl']:>3} {r.get('tm', 0):>3}")

    # Highlight best
    best = max(all_results, key=lambda x: x["ret"])
    print(f"\n  BEST: {best['name']} | Grade={best['grade']} | "
          f"Return={best['ret']*100:+.2f}% | WR={best['wr']:.1f}% | PF={best['pf']:.2f}")

    # Count by grade
    grades = {}
    for r in all_results:
        grades[r["grade"]] = grades.get(r["grade"], 0) + 1
    print(f"  Grade distribution: {grades}")


if __name__ == "__main__":
    main()
