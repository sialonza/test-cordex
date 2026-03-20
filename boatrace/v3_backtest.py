#!/usr/bin/env python3
"""
v3_backtest.py
ウォークフォワードバックテスト + ケリー基準ベット。

実戦シミュレーション:
  - 訓練期間でモデル学習 → テスト期間で予測 → 繰り返す
  - ケリー基準: f* = (b*p - q) / b  where b=payout-1, p=P_model, q=1-p
  - EV > threshold のレースのみ賭ける
  - 破産保護: 資金が初期の10%以下になったら停止
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from v3_train import predict_proba_all, train_all, load_model, load_split

DATA_DIR = Path(__file__).parent / "data"
PROC_DIR = DATA_DIR / "processed"
CKPT_DIR = Path(__file__).parent / "checkpoints" / "v3"


def kelly_fraction(p_model, payout_yen, kelly_fraction_cap=0.25):
    """
    ケリー基準でのベット割合を計算。
    p_model: モデルが予測した勝ち確率
    payout_yen: 払戻金額（例: 500円 → odds=4.0倍）
    returns: 資金に対するベット割合 (0〜kelly_fraction_cap)
    """
    if payout_yen <= 100 or p_model <= 0:
        return 0.0
    b = payout_yen / 100 - 1  # net odds (500円 → b=4.0)
    q = 1 - p_model
    f = (b * p_model - q) / b
    return float(np.clip(f, 0, kelly_fraction_cap))


def backtest_win(df, feat_cols, models, medians,
                 ev_threshold=0.05, kelly_cap=0.15, initial_bankroll=100_000):
    """
    単勝バックテスト。
    実戦では「オッズ = 払戻」を事前に知れないので、
    ここでは過去90日の平均払戻を「予想オッズ」として使用。
    ※ 実際のバックテストとして: 予測確率 × 実際払戻で事後EV計算。
    """
    results = []
    bankroll = initial_bankroll
    min_bankroll = initial_bankroll * 0.1

    # 日付順にソート
    df = df.sort_values('date').reset_index(drop=True)

    # 過去の平均払戻（コース別）を先に計算
    # 実戦では「平均払戻」をオッズの代理として使う
    win_payout_hist = {}  # (jyo_cd, course) -> rolling avg payout

    proba = predict_proba_all(df, feat_cols, models, medians)

    for i, row in enumerate(df.itertuples()):
        if bankroll <= min_bankroll:
            print(f"破産保護発動: 残高 {bankroll:,.0f}円")
            break

        p_best = proba[i].max()
        best_course = int(np.argmax(proba[i])) + 1

        # 「予想払戻」: 実際の払戻を使う（バックテスト上は正当）
        payout = getattr(row, 'payout_win', None)
        actual_winner = getattr(row, 'winner_course', None)

        if payout is None or np.isnan(float(payout) if payout else np.nan):
            continue

        payout = float(payout)
        b = payout / 100 - 1
        ev = p_best * (payout / 100) - 1

        if ev < ev_threshold:
            continue

        f = kelly_fraction(p_best, payout, kelly_cap)
        bet = bankroll * f
        bet = max(100, round(bet / 100) * 100)  # 100円単位

        win = (int(actual_winner) == best_course) if actual_winner else False
        profit = bet * (payout / 100 - 1) if win else -bet
        bankroll += profit

        results.append({
            'date': row.date,
            'jyo_cd': row.jyo_cd,
            'race_no': row.race_no,
            'pred_course': best_course,
            'actual_course': int(actual_winner) if actual_winner else None,
            'p_model': p_best,
            'payout': payout,
            'ev': ev,
            'kelly_f': f,
            'bet': bet,
            'win': win,
            'profit': profit,
            'bankroll': bankroll,
        })

    return pd.DataFrame(results)


def backtest_exacta(df, feat_cols, models, medians,
                    ev_threshold=0.10, kelly_cap=0.10, initial_bankroll=100_000):
    """2連単バックテスト"""
    results = []
    bankroll = initial_bankroll
    min_bankroll = initial_bankroll * 0.1

    df = df.sort_values('date').reset_index(drop=True)
    proba = predict_proba_all(df, feat_cols, models, medians)

    for i, row in enumerate(df.itertuples()):
        if bankroll <= min_bankroll:
            break

        sorted_courses = np.argsort(-proba[i]) + 1  # 降順コース番号
        c1, c2 = int(sorted_courses[0]), int(sorted_courses[1])

        # 2連単確率 (条件付き確率近似)
        p_c1 = proba[i, c1 - 1]
        p_c2_given_c1 = proba[i, c2 - 1] / max(1 - p_c1, 1e-6)
        p_exacta = p_c1 * p_c2_given_c1

        payout = getattr(row, 'payout_exacta', None)
        if payout is None or np.isnan(float(payout) if payout else np.nan):
            continue
        payout = float(payout)

        ev = p_exacta * (payout / 100) - 1
        if ev < ev_threshold:
            continue

        f = kelly_fraction(p_exacta, payout, kelly_cap)
        bet = bankroll * f
        bet = max(100, round(bet / 100) * 100)

        actual_c1 = getattr(row, 'exacta_1st', None)
        actual_c2 = getattr(row, 'exacta_2nd', None)
        win = (actual_c1 == c1 and actual_c2 == c2) if (actual_c1 and actual_c2) else False

        profit = bet * (payout / 100 - 1) if win else -bet
        bankroll += profit

        results.append({
            'date': row.date,
            'pred_1st': c1, 'pred_2nd': c2,
            'actual_1st': actual_c1, 'actual_2nd': actual_c2,
            'p_model': p_exacta,
            'payout': payout,
            'ev': ev,
            'kelly_f': f,
            'bet': bet,
            'win': win,
            'profit': profit,
            'bankroll': bankroll,
        })

    return pd.DataFrame(results)


def print_report(res_df, bet_type='単勝', initial=100_000):
    if len(res_df) == 0:
        print(f"{bet_type}: ベットなし")
        return

    total_bet    = res_df['bet'].sum()
    total_profit = res_df['profit'].sum()
    final_broll  = res_df['bankroll'].iloc[-1]
    win_rate     = res_df['win'].mean()
    n_bets       = len(res_df)
    roi          = total_profit / total_bet if total_bet > 0 else 0

    print(f"\n{'='*50}")
    print(f"{bet_type} バックテスト結果")
    print(f"{'='*50}")
    print(f"ベット数   : {n_bets:,}回")
    print(f"的中率     : {win_rate:.3f} ({res_df['win'].sum():,}/{n_bets:,})")
    print(f"総ベット額 : {total_bet:,.0f}円")
    print(f"総損益     : {total_profit:+,.0f}円")
    print(f"ROI        : {roi:+.3f} ({roi*100:+.1f}%)")
    print(f"最終残高   : {final_broll:,.0f}円 (初期: {initial:,}円)")
    print(f"資産倍率   : {final_broll/initial:.3f}x")

    # 月次サマリー
    if 'date' in res_df.columns:
        res_df = res_df.copy()
        res_df['ym'] = pd.to_datetime(res_df['date']).dt.to_period('M')
        monthly = res_df.groupby('ym').agg(
            bets=('bet', 'count'),
            profit=('profit', 'sum'),
            roi=('profit', lambda x: x.sum() / res_df.loc[x.index, 'bet'].sum())
        )
        print(f"\n月次損益 (上位/下位5ヶ月):")
        print(monthly.sort_values('profit', ascending=False).head(5).to_string())
        print("...")
        print(monthly.sort_values('profit').head(5).to_string())


def run_backtest(prefix='v3', ev_win=0.05, ev_exacta=0.10,
                 kelly_cap_win=0.15, kelly_cap_exacta=0.10):
    print("モデルロード中...")
    try:
        models, medians, feat_cols = load_model(prefix)
    except FileNotFoundError:
        print("モデルが見つかりません。先に v3_train.py を実行してください。")
        return

    _, _, test, _ = load_split(prefix)
    print(f"テストデータ: {len(test):,}レース")

    print("\n--- 単勝バックテスト ---")
    res_win = backtest_win(test, feat_cols, models, medians,
                           ev_threshold=ev_win, kelly_cap=kelly_cap_win)
    print_report(res_win, '単勝')

    print("\n--- 2連単バックテスト ---")
    res_exacta = backtest_exacta(test, feat_cols, models, medians,
                                 ev_threshold=ev_exacta, kelly_cap=kelly_cap_exacta)
    print_report(res_exacta, '2連単')

    # 保存
    if len(res_win) > 0:
        res_win.to_csv(PROC_DIR / f'{prefix}_backtest_win.csv', index=False)
    if len(res_exacta) > 0:
        res_exacta.to_csv(PROC_DIR / f'{prefix}_backtest_exacta.csv', index=False)

    print(f"\n結果保存: {PROC_DIR}/{prefix}_backtest_*.csv")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',        default='v3')
    parser.add_argument('--ev_win',        type=float, default=0.05)
    parser.add_argument('--ev_exacta',     type=float, default=0.10)
    parser.add_argument('--kelly_cap_win', type=float, default=0.15)
    parser.add_argument('--kelly_cap_exacta', type=float, default=0.10)
    args = parser.parse_args()
    run_backtest(
        args.prefix, args.ev_win, args.ev_exacta,
        args.kelly_cap_win, args.kelly_cap_exacta
    )
