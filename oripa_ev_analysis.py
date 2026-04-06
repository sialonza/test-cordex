"""
Online Oripa Expected Value Analysis Framework
================================================
Mathematical analysis of Japanese online oripa (trading card lotteries).

This module provides tools for:
- Computing EV for finite-pool and infinite-supply oripa
- Tracking conditional EV as cards are drawn from a finite pool
- Simulating the "last one" (ラストワン賞) mechanic
- Identifying +EV entry points via card-counting logic
- Monte Carlo simulation for variance/risk analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import copy


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PrizeTier:
    """A single prize tier in an oripa pool."""
    name: str
    market_value: float       # Current market value in yen
    count: int                # Number of copies in the pool (finite pool)
    probability: float = 0.0  # For infinite-supply model (overrides count)


@dataclass
class OripaPool:
    """Complete oripa definition."""
    name: str
    pack_price: float                     # Price per pack in yen
    total_packs: int                      # Total packs in pool (finite model)
    prize_tiers: list                     # list[PrizeTier]
    last_one_prize_value: float = 0.0     # ラストワン賞 value (0 = none)
    is_infinite: bool = False             # True = probability model, no depletion
    drawn: list = field(default_factory=list)  # Indices of drawn tiers (for tracking)

    @property
    def total_prize_cards(self) -> int:
        return sum(t.count for t in self.prize_tiers)

    @property
    def remaining_packs(self) -> int:
        return self.total_packs - len(self.drawn)


# ---------------------------------------------------------------------------
# Core EV Calculations
# ---------------------------------------------------------------------------

def compute_ev(pool: OripaPool) -> dict:
    """
    Compute the expected value of a single pack purchase.

    For a finite pool:
        EV = Σ (n_i / N) * v_i
    where n_i = count of prize tier i, N = total packs, v_i = market value.

    For an infinite pool:
        EV = Σ p_i * v_i
    where p_i = fixed probability of tier i.

    Returns dict with EV, RTP (return to player), and house edge.
    """
    if pool.is_infinite:
        ev = sum(t.probability * t.market_value for t in pool.prize_tiers)
    else:
        N = pool.total_packs
        ev = sum((t.count / N) * t.market_value for t in pool.prize_tiers)

    rtp = ev / pool.pack_price
    house_edge = 1.0 - rtp

    return {
        "expected_value": ev,
        "pack_price": pool.pack_price,
        "rtp": rtp,
        "rtp_percent": rtp * 100,
        "house_edge": house_edge,
        "house_edge_percent": house_edge * 100,
        "ev_per_yen": ev / pool.pack_price,
    }


def compute_conditional_ev(pool: OripaPool, drawn_tiers: list[int]) -> dict:
    """
    Compute conditional EV after specific cards have been drawn from a finite pool.

    Parameters
    ----------
    pool : OripaPool
        The original (full) pool definition.
    drawn_tiers : list[int]
        List of tier indices that have been drawn. E.g., [2, 2, 3, 0]
        means one draw from tier 0, two from tier 2, one from tier 3.

    Returns
    -------
    dict with conditional EV, remaining pool composition, and whether
    the pool is now +EV.
    """
    if pool.is_infinite:
        # Infinite pools have no conditional EV shift
        return compute_ev(pool)

    # Count how many of each tier have been drawn
    drawn_counts = {}
    for tier_idx in drawn_tiers:
        drawn_counts[tier_idx] = drawn_counts.get(tier_idx, 0) + 1

    total_drawn = len(drawn_tiers)
    remaining_total = pool.total_packs - total_drawn

    if remaining_total <= 0:
        return {"error": "Pool is exhausted"}

    # Compute remaining counts and conditional EV
    remaining_tiers = []
    conditional_ev = 0.0
    total_remaining_value = 0.0

    for i, tier in enumerate(pool.prize_tiers):
        drawn_from_tier = drawn_counts.get(i, 0)
        remaining_in_tier = tier.count - drawn_from_tier

        if remaining_in_tier < 0:
            return {"error": f"More drawn from tier {i} ({tier.name}) than exist"}

        remaining_tiers.append({
            "name": tier.name,
            "market_value": tier.market_value,
            "original_count": tier.count,
            "remaining": remaining_in_tier,
            "conditional_probability": remaining_in_tier / remaining_total,
        })

        tier_ev_contribution = (remaining_in_tier / remaining_total) * tier.market_value
        conditional_ev += tier_ev_contribution
        total_remaining_value += remaining_in_tier * tier.market_value

    # Last-one adjustment: if pool hasn't been claimed yet
    last_one_adjustment = 0.0
    if pool.last_one_prize_value > 0:
        last_one_adjustment = pool.last_one_prize_value / remaining_total

    total_ev = conditional_ev + last_one_adjustment
    rtp = total_ev / pool.pack_price

    return {
        "conditional_ev": conditional_ev,
        "last_one_ev_addition": last_one_adjustment,
        "total_ev_with_last_one": total_ev,
        "pack_price": pool.pack_price,
        "rtp": rtp,
        "rtp_percent": rtp * 100,
        "is_positive_ev": total_ev > pool.pack_price,
        "packs_remaining": remaining_total,
        "packs_drawn": total_drawn,
        "remaining_tiers": remaining_tiers,
        "total_remaining_value": total_remaining_value,
    }


def find_positive_ev_threshold(pool: OripaPool) -> dict:
    """
    For a finite pool, determine the theoretical point at which the pool
    becomes +EV, assuming the worst case (highest-value cards are still
    in the pool) and best case (lowest-value cards are still in the pool).

    This answers: "After how many low-value draws does the pool become +EV?"

    Strategy: simulate drawing only the lowest-value cards first and track
    when the conditional EV exceeds the pack price.
    """
    if pool.is_infinite:
        return {"result": "Infinite pools never change EV -- no threshold exists."}

    # Sort tiers by market value (ascending)
    indexed_tiers = [(i, t) for i, t in enumerate(pool.prize_tiers)]
    indexed_tiers.sort(key=lambda x: x[1].market_value)

    # Best case: all low-value cards drawn first, high-value remain
    drawn = []
    threshold_best = None

    for tier_idx, tier in indexed_tiers:
        for _ in range(tier.count):
            drawn.append(tier_idx)
            result = compute_conditional_ev(pool, drawn)
            if "error" in result:
                break
            if result["total_ev_with_last_one"] > pool.pack_price and threshold_best is None:
                threshold_best = {
                    "draws_needed": len(drawn),
                    "packs_remaining": result["packs_remaining"],
                    "conditional_ev": result["total_ev_with_last_one"],
                    "rtp_percent": result["rtp_percent"],
                }
                break
        if threshold_best:
            break

    # Worst case: high-value cards drawn first (EV never improves)
    drawn_worst = []
    indexed_tiers_desc = sorted(
        [(i, t) for i, t in enumerate(pool.prize_tiers)],
        key=lambda x: x[1].market_value,
        reverse=True,
    )
    threshold_worst = None
    for tier_idx, tier in indexed_tiers_desc:
        for _ in range(tier.count):
            drawn_worst.append(tier_idx)
            result = compute_conditional_ev(pool, drawn_worst)
            if "error" in result:
                break
            if result["total_ev_with_last_one"] > pool.pack_price:
                threshold_worst = {
                    "draws_needed": len(drawn_worst),
                    "packs_remaining": result["packs_remaining"],
                    "conditional_ev": result["total_ev_with_last_one"],
                    "rtp_percent": result["rtp_percent"],
                }
                break
        if threshold_worst:
            break

    return {
        "best_case_threshold": threshold_best,
        "worst_case_threshold": threshold_worst,
        "interpretation": (
            "best_case = only trash cards drawn before you enter; "
            "worst_case = top prizes already pulled before you enter. "
            "If worst_case is None, the pool can never go +EV by draw "
            "depletion alone (house edge too large)."
        ),
    }


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

def simulate_oripa(pool: OripaPool, n_simulations: int = 100_000,
                   packs_per_sim: int = 1, seed: int = 42) -> dict:
    """
    Monte Carlo simulation of buying packs_per_sim packs from the oripa.

    For finite pools, simulates drawing without replacement.
    For infinite pools, simulates independent draws.

    Returns distribution statistics for total value received.
    """
    rng = np.random.default_rng(seed)

    # Build the full pool as an array of values
    if pool.is_infinite:
        probs = np.array([t.probability for t in pool.prize_tiers])
        values = np.array([t.market_value for t in pool.prize_tiers])

        # Draw tier indices for all sims at once
        tier_draws = rng.choice(
            len(pool.prize_tiers),
            size=(n_simulations, packs_per_sim),
            p=probs,
        )
        total_values = values[tier_draws].sum(axis=1)
    else:
        # Build full pool
        full_pool = []
        for tier in pool.prize_tiers:
            full_pool.extend([tier.market_value] * tier.count)
        full_pool = np.array(full_pool)

        if packs_per_sim > len(full_pool):
            return {"error": "packs_per_sim exceeds pool size"}

        total_values = np.empty(n_simulations)
        for i in range(n_simulations):
            drawn_indices = rng.choice(len(full_pool), size=packs_per_sim, replace=False)
            total_values[i] = full_pool[drawn_indices].sum()

    total_cost = packs_per_sim * pool.pack_price
    profits = total_values - total_cost

    return {
        "n_simulations": n_simulations,
        "packs_per_sim": packs_per_sim,
        "total_cost": total_cost,
        "mean_value": float(np.mean(total_values)),
        "median_value": float(np.median(total_values)),
        "std_value": float(np.std(total_values)),
        "mean_profit": float(np.mean(profits)),
        "median_profit": float(np.median(profits)),
        "prob_profit": float(np.mean(profits > 0)),
        "prob_break_even": float(np.mean(profits >= 0)),
        "percentile_5": float(np.percentile(total_values, 5)),
        "percentile_25": float(np.percentile(total_values, 25)),
        "percentile_75": float(np.percentile(total_values, 75)),
        "percentile_95": float(np.percentile(total_values, 95)),
        "max_value": float(np.max(total_values)),
        "min_value": float(np.min(total_values)),
        "worst_loss": float(np.min(profits)),
        "best_win": float(np.max(profits)),
    }


# ---------------------------------------------------------------------------
# Pool Analysis / Red Flag Detection
# ---------------------------------------------------------------------------

def analyze_pool_structure(pool: OripaPool) -> dict:
    """
    Analyze an oripa pool for structural properties and red flags.
    """
    ev_data = compute_ev(pool)

    # Concentration of value: what % of total pool value is in the top prize?
    total_value = sum(t.count * t.market_value for t in pool.prize_tiers)
    top_tier = max(pool.prize_tiers, key=lambda t: t.market_value)
    top_tier_total_value = top_tier.count * top_tier.market_value
    value_concentration = top_tier_total_value / total_value if total_value > 0 else 0

    # What fraction of packs are "winners" (value >= pack price)?
    if pool.is_infinite:
        winner_prob = sum(
            t.probability for t in pool.prize_tiers
            if t.market_value >= pool.pack_price
        )
    else:
        winner_count = sum(
            t.count for t in pool.prize_tiers
            if t.market_value >= pool.pack_price
        )
        winner_prob = winner_count / pool.total_packs

    # Total revenue vs total prize value
    total_revenue = pool.total_packs * pool.pack_price
    operator_margin = (total_revenue - total_value) / total_revenue if total_revenue > 0 else 0

    # Red flags
    red_flags = []
    if ev_data["rtp"] < 0.30:
        red_flags.append(f"CRITICAL: RTP is only {ev_data['rtp_percent']:.1f}% -- extremely exploitative")
    elif ev_data["rtp"] < 0.50:
        red_flags.append(f"WARNING: RTP is {ev_data['rtp_percent']:.1f}% -- poor value")

    if value_concentration > 0.50:
        red_flags.append(
            f"High value concentration: {value_concentration*100:.1f}% of pool value "
            f"is in {top_tier.name}. Most packs will be worthless."
        )

    if winner_prob < 0.05:
        red_flags.append(
            f"Only {winner_prob*100:.2f}% chance of getting a card worth >= pack price"
        )

    if pool.is_infinite:
        red_flags.append(
            "Infinite supply model -- probabilities never improve. "
            "This is mathematically identical to a slot machine."
        )

    return {
        "ev_analysis": ev_data,
        "total_pool_value": total_value,
        "total_revenue": total_revenue,
        "operator_margin_percent": operator_margin * 100,
        "top_prize": {
            "name": top_tier.name,
            "value": top_tier.market_value,
            "count": top_tier.count,
            "value_concentration_percent": value_concentration * 100,
        },
        "winner_probability": winner_prob,
        "winner_probability_percent": winner_prob * 100,
        "red_flags": red_flags,
    }


# ---------------------------------------------------------------------------
# Demonstration: Realistic Oripa Example
# ---------------------------------------------------------------------------

def demo_realistic_oripa():
    """
    Demonstrate analysis with a realistic oripa pool.
    Based on typical structures seen on DOPA/clove/Iris type platforms.

    Example: A 1,000-yen Pokemon oripa with 100 total packs.
    """
    # ===================================================================
    # REALISTIC POOL: Operator targets ~55% margin (45% RTP).
    # ===================================================================
    # Total revenue = 1000 * 500 = 500,000 yen
    # Total prize value = ~225,000 yen (~45% RTP)
    #
    # KEY INSIGHT: The operator buys cards at wholesale/auction prices
    # which are LOWER than market value. So their actual cost might be
    # 60-70% of the market values listed here. But RTP is calculated
    # from the MARKET VALUE the player receives, not operator cost.
    #
    # This pool is modeled on real DOPA/clove structures:
    # - One huge chase card (the "bait")
    # - A few mid-tier prizes
    # - Massive majority of trash cards worth a fraction of pack price
    pool = OripaPool(
        name="Pokemon EX Special Oripa (1000 yen x 500 packs)",
        pack_price=1000,
        total_packs=500,
        last_one_prize_value=30000,  # ラストワン: card worth 30,000 yen
        prize_tiers=[
            PrizeTier(name="PSA10 Charizard ex SAR", market_value=150000, count=1),
            PrizeTier(name="Pikachu ex SR",          market_value=20000,  count=1),
            PrizeTier(name="Mew ex SAR",             market_value=8000,   count=2),
            PrizeTier(name="Assorted SR cards",      market_value=3000,   count=5),
            PrizeTier(name="Assorted RR/AR decent",  market_value=1000,   count=15),
            PrizeTier(name="Assorted holos",         market_value=300,    count=40),
            PrizeTier(name="Bulk commons/energy",    market_value=50,     count=436),
        ],
        # Total value: 150000 + 20000 + 16000 + 15000 + 15000 + 12000 + 21800
        #            = 249,800 yen
        # Revenue: 500,000 yen
        # RTP = 249,800/500,000 = 49.96% (before last-one)
        # With last-one: (249,800 + 30,000)/500,000 = 55.96%
    )

    print("=" * 70)
    print(f"ORIPA: {pool.name}")
    print("=" * 70)

    # 1. Basic EV
    print("\n--- Initial EV Analysis ---")
    ev = compute_ev(pool)
    for k, v in ev.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # 2. Pool structure analysis
    print("\n--- Pool Structure Analysis ---")
    analysis = analyze_pool_structure(pool)
    print(f"  Total pool value:        {analysis['total_pool_value']:,.0f} yen")
    print(f"  Total revenue:           {analysis['total_revenue']:,.0f} yen")
    print(f"  Operator margin:         {analysis['operator_margin_percent']:.1f}%")
    print(f"  Winner probability:      {analysis['winner_probability_percent']:.2f}%")
    print(f"  Top prize concentration: {analysis['top_prize']['value_concentration_percent']:.1f}%")
    for flag in analysis["red_flags"]:
        print(f"  [FLAG] {flag}")

    # 3. Conditional EV after some draws
    print("\n--- Conditional EV: After 200 Bulk Draws ---")
    # Simulate 200 bulk (tier index 6) cards being drawn
    drawn = [6] * 200
    cond = compute_conditional_ev(pool, drawn)
    print(f"  Packs remaining:   {cond['packs_remaining']}")
    print(f"  Conditional EV:    {cond['conditional_ev']:.2f} yen")
    print(f"  + Last One adj:    {cond['last_one_ev_addition']:.2f} yen")
    print(f"  Total EV:          {cond['total_ev_with_last_one']:.2f} yen")
    print(f"  RTP:               {cond['rtp_percent']:.1f}%")
    print(f"  Is +EV?            {cond['is_positive_ev']}")

    # 4. Find +EV threshold
    print("\n--- +EV Threshold Analysis ---")
    threshold = find_positive_ev_threshold(pool)
    if threshold["best_case_threshold"]:
        t = threshold["best_case_threshold"]
        print(f"  Best case: +EV after {t['draws_needed']} low-value draws")
        print(f"    Packs remaining: {t['packs_remaining']}")
        print(f"    Conditional EV:  {t['conditional_ev']:.2f} yen")
        print(f"    RTP:             {t['rtp_percent']:.1f}%")
    else:
        print("  Best case: Pool never reaches +EV")

    if threshold["worst_case_threshold"]:
        t = threshold["worst_case_threshold"]
        print(f"  Worst case: +EV after {t['draws_needed']} high-value draws")
    else:
        print("  Worst case: Pool never reaches +EV (expected)")

    # 5. Monte Carlo
    print("\n--- Monte Carlo Simulation (100k trials, 1 pack each) ---")
    sim = simulate_oripa(pool, n_simulations=100_000, packs_per_sim=1)
    print(f"  Mean value:          {sim['mean_value']:,.0f} yen")
    print(f"  Median value:        {sim['median_value']:,.0f} yen")
    print(f"  Mean profit:         {sim['mean_profit']:,.0f} yen")
    print(f"  Median profit:       {sim['median_profit']:,.0f} yen")
    print(f"  P(profit > 0):       {sim['prob_profit']*100:.2f}%")
    print(f"  P(break even):       {sim['prob_break_even']*100:.2f}%")
    print(f"  5th percentile:      {sim['percentile_5']:,.0f} yen")
    print(f"  95th percentile:     {sim['percentile_95']:,.0f} yen")
    print(f"  Best outcome:        {sim['best_win']:,.0f} yen profit")
    print(f"  Worst outcome:       {sim['worst_loss']:,.0f} yen loss")

    # 6. Monte Carlo: buying 10 packs
    print("\n--- Monte Carlo: 10 Packs per Session ---")
    sim10 = simulate_oripa(pool, n_simulations=100_000, packs_per_sim=10)
    print(f"  Total cost:          {sim10['total_cost']:,.0f} yen")
    print(f"  Mean value:          {sim10['mean_value']:,.0f} yen")
    print(f"  Mean profit:         {sim10['mean_profit']:,.0f} yen")
    print(f"  P(profit > 0):       {sim10['prob_profit']*100:.2f}%")
    print(f"  5th percentile:      {sim10['percentile_5']:,.0f} yen")
    print(f"  95th percentile:     {sim10['percentile_95']:,.0f} yen")

    # 7. Comparison: What if same pool were infinite supply?
    print("\n--- Comparison: Same Odds as Infinite Supply ---")
    inf_pool = OripaPool(
        name="Same pool, infinite supply",
        pack_price=1000,
        total_packs=500,  # Not used in infinite mode
        is_infinite=True,
        prize_tiers=[
            PrizeTier(name="PSA10 Charizard ex SAR", market_value=150000, count=1, probability=0.002),
            PrizeTier(name="Pikachu ex SR",          market_value=20000,  count=1, probability=0.002),
            PrizeTier(name="Mew ex SAR",             market_value=8000,   count=2, probability=0.004),
            PrizeTier(name="Assorted SR cards",      market_value=3000,   count=5, probability=0.010),
            PrizeTier(name="Assorted RR/AR decent",  market_value=1000,   count=15, probability=0.030),
            PrizeTier(name="Assorted holos",         market_value=300,    count=40, probability=0.080),
            PrizeTier(name="Bulk commons/energy",    market_value=50,     count=436, probability=0.872),
        ],
    )
    ev_inf = compute_ev(inf_pool)
    print(f"  EV:    {ev_inf['expected_value']:.2f} yen (same as finite initial EV)")
    print(f"  RTP:   {ev_inf['rtp_percent']:.1f}%")
    print(f"  Key difference: This EV NEVER changes. No card-counting possible.")

    return pool, analysis, threshold


def demo_ev_over_time(pool: OripaPool):
    """
    Show how conditional EV evolves as the cheapest cards are drawn first.
    This is the "card counting" scenario -- you're watching a transparent pool
    and only buying when the EV is favorable.
    """
    print("\n" + "=" * 70)
    print("EV EVOLUTION: Watching a Transparent Pool")
    print("(Assumes cheapest cards are drawn by others first)")
    print("=" * 70)

    # Sort tiers by value ascending
    indexed_tiers = [(i, t) for i, t in enumerate(pool.prize_tiers)]
    indexed_tiers.sort(key=lambda x: x[1].market_value)

    drawn = []
    N = pool.total_packs
    # Generate checkpoints at 0%, 10%, 20%, ..., 90%, and second-to-last
    checkpoints = [int(N * pct / 100) for pct in range(0, 100, 10)]

    print(f"\n  {'Drawn':>6} | {'Remaining':>9} | {'Cond. EV':>10} | {'RTP':>7} | {'Status':>10}")
    print("  " + "-" * 55)

    tier_queue = []
    for tier_idx, tier in indexed_tiers:
        tier_queue.extend([tier_idx] * tier.count)

    for draw_num in range(len(tier_queue)):
        drawn.append(tier_queue[draw_num])

        if draw_num in checkpoints or draw_num == len(tier_queue) - 2:
            result = compute_conditional_ev(pool, drawn)
            if "error" in result:
                break
            ev = result["total_ev_with_last_one"]
            rtp = result["rtp_percent"]
            status = "+EV !" if result["is_positive_ev"] else "-EV"
            print(f"  {draw_num+1:>6} | {result['packs_remaining']:>9} | {ev:>10,.0f} | {rtp:>6.1f}% | {status:>10}")


if __name__ == "__main__":
    pool, analysis, threshold = demo_realistic_oripa()
    demo_ev_over_time(pool)
