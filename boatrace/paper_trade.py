#!/usr/bin/env python3
"""
paper_trade.py
テストデータを使ってペーパートレード（仮想ベット）をシミュレートする。

【運用ルール（全て設定可能）】
  - EV 閾値          : ev_threshold       (例: 1.0)  EV≥閾値のレースのみベット
  - 1日最大ベット数  : max_bets_per_day   (例: 5)    1日に何レースまでベットするか
  - 日次損失上限     : daily_loss_limit   (例: -2000) 1日の損失がこの額を超えたら当日ストップ
  - 連敗ストップ     : max_consec_losses  (例: 10)   連続N回外れたら全停止
  - 総損失上限       : total_loss_limit   (例: -20000)累積損失がこの額を下回ったら全停止
  - ベット単位       : bet_unit           (例: 100円)
  - 初期資金         : initial_capital    (例: 50000円)

【出力】
  checkpoints/real55k/paper_trade_journal.csv   # 日次・レース別詳細ログ
  checkpoints/real55k/paper_trade_summary.json  # サマリー統計
  （標準出力）日次損益テーブル・ルール発動履歴
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from dataclasses import dataclass, field, asdict
from typing import List, Optional

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "real55k")

WINNER_THRESHOLD = 0.25


# ── 設定 ─────────────────────────────────────────────────────────────────────

@dataclass
class RuleConfig:
    """運用ルール設定。すべてここで管理する。"""
    ev_threshold:       float = 1.0    # EV 閾値（1.0 = 期待値トントン以上）
    max_bets_per_day:   int   = 5      # 1日最大ベット数
    daily_loss_limit:   int   = -2000  # 日次損失上限 (円, 負値)
    max_consec_losses:  int   = 10     # 連敗ストップ回数
    total_loss_limit:   int   = -20000 # 総損失上限 (円, 負値)
    bet_unit:           int   = 100    # 1ベット単位 (円)
    initial_capital:    int   = 50000  # 初期資金 (円)


# ── 状態管理 ─────────────────────────────────────────────────────────────────

@dataclass
class TradingState:
    """ペーパートレード中のステート。"""
    capital:            float        = 0.0
    consec_losses:      int          = 0
    total_bets:         int          = 0
    total_hits:         int          = 0
    total_payout:       float        = 0.0
    max_drawdown:       float        = 0.0
    peak_capital:       float        = 0.0
    stopped:            bool         = False
    stop_reason:        str          = ""
    rule_triggers:      List[dict]   = field(default_factory=list)


# ── アセット読み込み ──────────────────────────────────────────────────────────

def load_assets():
    model_win = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_win.txt"))
    model_2nd = lgb.Booster(model_file=os.path.join(CKPT_DIR, "lgbm_2nd.txt"))

    with open(os.path.join(DATA_DIR, "feature_cols.txt")) as f:
        feature_cols = [l.strip() for l in f if l.strip()]

    stacking_path = os.path.join(CKPT_DIR, "stacking_feature_cols.txt")
    stacking_cols = None
    if os.path.exists(stacking_path):
        with open(stacking_path) as f:
            stacking_cols = [l.strip() for l in f if l.strip()]

    calibrators = {}
    for mname in ("lgbm_win", "lgbm_2nd"):
        for method in ("isotonic", "platt"):
            p = os.path.join(CKPT_DIR, f"{mname}_{method}.pkl")
            if os.path.exists(p) and mname not in calibrators:
                calibrators[mname] = joblib.load(p)
                print(f"  キャリブレーター: {mname} ({method})")
                break

    tm_path = os.path.join(CKPT_DIR, "transition_matrix.json")
    trans_matrix = {}
    if os.path.exists(tm_path):
        with open(tm_path) as f:
            trans_matrix = json.load(f)

    with open(os.path.join(CKPT_DIR, "thresholds.json")) as f:
        thresholds = json.load(f)

    payout_path = os.path.join(RAW_DIR, "race_payouts.csv")
    if os.path.exists(payout_path):
        payouts = pd.read_csv(payout_path, dtype={"jyo_cd": str})
        payouts["jyo_cd"] = payouts["jyo_cd"].str.zfill(2)
        payouts["date"]   = payouts["date"].astype(str)
    else:
        payouts = None
        print("警告: race_payouts.csv が見つかりません（払戻データなし）")

    return (model_win, model_2nd, feature_cols, stacking_cols,
            calibrators, trans_matrix, thresholds, payouts)


def apply_calibrator(cal, raw_prob):
    if cal is None:
        return raw_prob
    if hasattr(cal, "predict_proba"):
        return cal.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
    return np.clip(cal.predict(raw_prob), 1e-7, 1 - 1e-7)


# ── 推論 ─────────────────────────────────────────────────────────────────────

def build_predictions(test_df, model_win, model_2nd, feature_cols, stacking_cols,
                      calibrators, trans_matrix, thresholds, payouts):
    """
    テストデータ全体を一度推論し、レース単位の予測 DataFrame を返す。
    払戻データをマージし、EV も計算する。
    """
    df = test_df.copy()
    df["date"] = df["date"].astype(str)

    avail_win = [c for c in feature_cols if c in df.columns]
    raw_win   = model_win.predict(df[avail_win].values,
                                   num_iteration=model_win.best_iteration)
    df["prob_win_raw"] = raw_win

    grp = df.groupby(["date", "jyo_cd", "race_no"])["prob_win_raw"]
    df["prob_win_max_in_race"] = grp.transform("max")
    df["prob_win_rank"]        = grp.rank(ascending=False)
    df["is_likely_winner"]     = (df["prob_win_raw"] >= WINNER_THRESHOLD).astype(int)

    if stacking_cols is not None:
        avail_2nd = [c for c in stacking_cols if c in df.columns]
    else:
        avail_2nd = avail_win + ["prob_win_raw", "prob_win_max_in_race",
                                  "prob_win_rank", "is_likely_winner"]
        avail_2nd = [c for c in avail_2nd if c in df.columns]

    raw_2nd = model_2nd.predict(df[avail_2nd].values,
                                 num_iteration=model_2nd.best_iteration)

    df["prob_win"] = apply_calibrator(calibrators.get("lgbm_win"), raw_win)
    df["prob_2nd"] = apply_calibrator(calibrators.get("lgbm_2nd"), raw_2nd)

    thr_win = thresholds.get("lgbm_win", 0.5)

    rows = []
    for (date, jyo_cd, race_no), g in df.groupby(["date", "jyo_cd", "race_no"]):
        if len(g) < 2:
            continue

        win_cands = g[g["prob_win"] >= thr_win]
        if len(win_cands) == 0:
            win_cands = g
        idx1 = win_cands["prob_win"].idxmax()
        c1   = int(g.loc[idx1, "course"])
        p1   = float(g.loc[idx1, "prob_win"])

        rest = g[g.index != idx1]
        idx2 = rest["prob_2nd"].idxmax()
        c2   = int(rest.loc[idx2, "course"])

        trans_key   = f"{c1}-{c2}"
        trans_prob  = trans_matrix.get(trans_key, 0.2)
        prob_exacta = p1 * trans_prob

        actual   = g.set_index("rank")["course"].to_dict()
        true_1st = int(actual.get(1, -1))
        true_2nd = int(actual.get(2, -1))
        hit      = (c1 == true_1st and c2 == true_2nd)

        rows.append({
            "date":        str(date),
            "jyo_cd":      str(jyo_cd).zfill(2),
            "race_no":     int(race_no),
            "pred_1st":    c1,
            "pred_2nd":    c2,
            "prob_win":    round(p1, 5),
            "prob_exacta": round(prob_exacta, 6),
            "true_1st":    true_1st,
            "true_2nd":    true_2nd,
            "hit":         hit,
        })

    pred_df = pd.DataFrame(rows).sort_values(["date", "jyo_cd", "race_no"])

    # 払戻マージ
    if payouts is not None:
        pred_df = pred_df.merge(
            payouts[["date", "jyo_cd", "race_no", "payout_2nd"]],
            on=["date", "jyo_cd", "race_no"], how="left"
        )
    else:
        pred_df["payout_2nd"] = np.nan

    # EV 計算 (払戻があるレースのみ)
    pred_df["ev"] = pred_df["prob_exacta"] * pred_df["payout_2nd"] / 100
    pred_df["ev"] = pred_df["ev"].fillna(0.0)

    # 損益分岐オッズ (円)
    pred_df["breakeven_odds"] = (100 / pred_df["prob_exacta"].clip(lower=1e-6)).round(0)

    return pred_df


# ── ペーパートレード シミュレーション ─────────────────────────────────────────

def simulate(pred_df: pd.DataFrame, cfg: RuleConfig):
    """
    pred_df を時系列順にスキャンし、運用ルールを適用してベット結果を記録する。

    Returns
    -------
    journal : pd.DataFrame   レース別ジャーナル
    daily   : pd.DataFrame   日次サマリー
    state   : TradingState   最終ステート
    """
    state        = TradingState(
        capital      = cfg.initial_capital,
        peak_capital = cfg.initial_capital,
    )
    journal_rows = []

    # 日次ループ
    for date, day_df in pred_df.groupby("date"):
        if state.stopped:
            break

        day_bets     = 0
        day_profit   = 0
        day_payout   = 0
        day_stopped  = False  # 当日の日次ルールによるストップフラグ

        # EV 降順でソートして上位からベット
        day_df_sorted = day_df.sort_values("ev", ascending=False)

        for _, row in day_df_sorted.iterrows():
            # ─── グローバル停止チェック ────────────────────────────────
            if state.stopped:
                break

            # ─── 日次ルールチェック ────────────────────────────────────
            if day_stopped:
                break

            if day_bets >= cfg.max_bets_per_day:
                break

            if day_profit <= cfg.daily_loss_limit:
                day_stopped = True
                state.rule_triggers.append({
                    "date":   date,
                    "rule":   "daily_loss_limit",
                    "detail": f"{day_profit:+,}円 ≤ {cfg.daily_loss_limit:,}円",
                })
                break

            # ─── EV フィルタ ───────────────────────────────────────────
            if row["ev"] < cfg.ev_threshold:
                break  # EV 降順のためここ以降はすべて閾値未満

            # 払戻不明ならスキップ
            if pd.isna(row.get("payout_2nd")):
                continue

            # ─── ベット実行 ────────────────────────────────────────────
            hit     = bool(row["hit"])
            payout  = float(row["payout_2nd"]) if hit else 0.0
            profit  = payout - cfg.bet_unit

            state.capital        += profit
            day_bets             += 1
            day_profit           += profit
            day_payout           += payout
            state.total_bets     += 1
            state.total_payout   += payout

            if hit:
                state.total_hits  += 1
                state.consec_losses = 0
            else:
                state.consec_losses += 1

            # ドローダウン更新
            state.peak_capital = max(state.peak_capital, state.capital)
            dd = (state.peak_capital - state.capital) / state.peak_capital
            state.max_drawdown = max(state.max_drawdown, dd)

            journal_rows.append({
                "date":          date,
                "jyo_cd":        row["jyo_cd"],
                "race_no":       row["race_no"],
                "pred_1st":      row["pred_1st"],
                "pred_2nd":      row["pred_2nd"],
                "prob_exacta":   row["prob_exacta"],
                "ev":            row["ev"],
                "breakeven":     row["breakeven_odds"],
                "payout_2nd":    row["payout_2nd"],
                "hit":           hit,
                "profit":        profit,
                "capital":       state.capital,
                "consec_losses": state.consec_losses,
            })

            # ─── ベット後ルールチェック ────────────────────────────────
            if state.consec_losses >= cfg.max_consec_losses:
                state.stopped     = True
                state.stop_reason = "max_consec_losses"
                state.rule_triggers.append({
                    "date":   date,
                    "rule":   "max_consec_losses",
                    "detail": f"連敗 {state.consec_losses} 回",
                })
                break

            if (state.capital - cfg.initial_capital) <= cfg.total_loss_limit:
                state.stopped     = True
                state.stop_reason = "total_loss_limit"
                state.rule_triggers.append({
                    "date":   date,
                    "rule":   "total_loss_limit",
                    "detail": f"累積損益 {state.capital - cfg.initial_capital:+,}円",
                })
                break

    journal = pd.DataFrame(journal_rows)

    # 日次サマリー生成
    if not journal.empty:
        daily = journal.groupby("date").agg(
            n_bets    = ("hit", "count"),
            n_hits    = ("hit", "sum"),
            profit    = ("profit", "sum"),
            payout    = ("payout_2nd", "sum"),
            capital   = ("capital", "last"),
        ).reset_index()
        daily["cum_profit"] = daily["profit"].cumsum()
        daily["roi_day"]    = daily["profit"] / (daily["n_bets"] * cfg.bet_unit)
    else:
        daily = pd.DataFrame()

    return journal, daily, state


# ── レポート表示 ──────────────────────────────────────────────────────────────

def print_report(journal, daily, state, cfg):
    total_profit   = state.capital - cfg.initial_capital
    roi_total      = total_profit / (state.total_bets * cfg.bet_unit) if state.total_bets else 0
    hit_rate       = state.total_hits / state.total_bets if state.total_bets else 0

    print("\n" + "=" * 70)
    print("ペーパートレード シミュレーション結果")
    print("=" * 70)
    print(f"  初期資金:         {cfg.initial_capital:>10,}円")
    print(f"  最終資金:         {state.capital:>10,.0f}円")
    print(f"  累積損益:         {total_profit:>+10,.0f}円")
    print(f"  ROI (全ベット):   {roi_total:>+10.1%}")
    print(f"  総ベット数:       {state.total_bets:>10,}")
    print(f"  的中数:           {state.total_hits:>10,}")
    print(f"  的中率:           {hit_rate:>10.2%}")
    print(f"  最大ドローダウン: {state.max_drawdown:>10.2%}")
    if state.stopped:
        print(f"  ★ 停止理由: {state.stop_reason}")

    # ルール発動履歴
    if state.rule_triggers:
        print(f"\n【運用ルール発動履歴】  ({len(state.rule_triggers)} 件)")
        for t in state.rule_triggers[-10:]:  # 最新10件
            print(f"  {t['date']}  [{t['rule']}]  {t['detail']}")

    # 日次サマリー (直近20日)
    if not daily.empty:
        print(f"\n【日次損益 (最新20日)】")
        print(f"  {'日付':>10}  {'ベット':>5}  {'的中':>5}  {'損益':>9}  {'累積':>10}  {'残高':>10}")
        print(f"  {'─' * 58}")
        for _, row in daily.tail(20).iterrows():
            sign = "+" if row["profit"] >= 0 else ""
            print(
                f"  {row['date']:>10}  {int(row['n_bets']):>5}  {int(row['n_hits']):>5}  "
                f"{sign}{row['profit']:>7,.0f}円  "
                f"{row['cum_profit']:>+8,.0f}円  "
                f"{row['capital']:>8,.0f}円"
            )


# ── サマリー保存 ──────────────────────────────────────────────────────────────

def build_summary(journal, daily, state, cfg):
    total_profit = state.capital - cfg.initial_capital
    total_cost   = state.total_bets * cfg.bet_unit

    monthly_roi = None
    if not daily.empty:
        daily["yearmonth"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)
        monthly = daily.groupby("yearmonth").agg(
            n_bets  = ("n_bets", "sum"),
            profit  = ("profit", "sum"),
        )
        monthly["roi"] = monthly["profit"] / (monthly["n_bets"] * cfg.bet_unit)
        monthly_roi = {
            "mean":   round(float(monthly["roi"].mean()), 5),
            "std":    round(float(monthly["roi"].std()), 5),
            "n_positive_months": int((monthly["roi"] > 0).sum()),
            "n_months": len(monthly),
        }

    summary = {
        "config":          asdict(cfg),
        "initial_capital": cfg.initial_capital,
        "final_capital":   round(state.capital, 2),
        "total_profit":    round(total_profit, 2),
        "roi_total":       round(total_profit / total_cost, 5) if total_cost else None,
        "total_bets":      state.total_bets,
        "total_hits":      state.total_hits,
        "hit_rate":        round(state.total_hits / state.total_bets, 5) if state.total_bets else None,
        "max_drawdown":    round(state.max_drawdown, 5),
        "stopped":         state.stopped,
        "stop_reason":     state.stop_reason,
        "n_rule_triggers": len(state.rule_triggers),
        "rule_triggers":   state.rule_triggers,
        "monthly_roi":     monthly_roi,
    }
    return summary


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ペーパートレード シミュレーター")
    print("=" * 70)

    cfg = RuleConfig(
        ev_threshold      = 1.0,
        max_bets_per_day  = 5,
        daily_loss_limit  = -2000,
        max_consec_losses = 10,
        total_loss_limit  = -20000,
        bet_unit          = 100,
        initial_capital   = 50000,
    )

    print(f"\n【運用ルール設定】")
    print(f"  EV 閾値:          {cfg.ev_threshold}")
    print(f"  1日最大ベット数:  {cfg.max_bets_per_day}")
    print(f"  日次損失上限:     {cfg.daily_loss_limit:,}円")
    print(f"  連敗ストップ:     {cfg.max_consec_losses}連敗")
    print(f"  総損失上限:       {cfg.total_loss_limit:,}円")
    print(f"  ベット単位:       {cfg.bet_unit:,}円")
    print(f"  初期資金:         {cfg.initial_capital:,}円")

    print("\nアセット読み込み中...")
    (model_win, model_2nd, feature_cols, stacking_cols,
     calibrators, trans_matrix, thresholds, payouts) = load_assets()

    print("テストデータ読み込み中...")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"テストサンプル: {len(test):,}  ({test['date'].min()} ~ {test['date'].max()})")

    print("予測生成中...")
    pred_df = build_predictions(
        test, model_win, model_2nd, feature_cols, stacking_cols,
        calibrators, trans_matrix, thresholds, payouts
    )
    print(f"予測レース数: {len(pred_df):,}")
    ev_filtered = (pred_df["ev"] >= cfg.ev_threshold).sum()
    print(f"EV≥{cfg.ev_threshold} のレース数: {ev_filtered:,} ({ev_filtered/len(pred_df):.1%})")

    print("\nシミュレーション実行中...")
    journal, daily, state = simulate(pred_df, cfg)

    print_report(journal, daily, state, cfg)

    # 保存
    journal_path = os.path.join(CKPT_DIR, "paper_trade_journal.csv")
    summary_path = os.path.join(CKPT_DIR, "paper_trade_summary.json")

    if not journal.empty:
        journal.to_csv(journal_path, index=False, encoding="utf-8-sig")
        print(f"\nジャーナル CSV: {journal_path}")

    summary = build_summary(journal, daily, state, cfg)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"サマリー JSON:  {summary_path}")

    print("\nペーパートレード完了!")


if __name__ == "__main__":
    main()
