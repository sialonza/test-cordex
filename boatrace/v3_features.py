#!/usr/bin/env python3
"""
v3_features.py
実戦で勝つための特徴量エンジニアリング。
- 選手ごとのコース別ローリング勝率
- モーター相対性能
- スタートタイミング統計
- 会場・コースバイアス
- 天候補正
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(exist_ok=True)

PAYOUT_TAKE_RATE = 0.25  # 舟券控除率（約25%）


def load_data(results_file=None, payouts_file=None):
    """結果と払戻データをロード・マージ"""
    # どちらのファイルを使うか自動選択
    if results_file is None:
        for fname in ['race_results_full.csv', 'race_results_real.csv']:
            path = RAW_DIR / fname
            if path.exists():
                results_file = path
                break
    if payouts_file is None:
        for fname in ['race_payouts_full.csv', 'race_payouts_real.csv']:
            path = RAW_DIR / fname
            if path.exists():
                payouts_file = path
                break

    print(f"結果: {results_file}")
    print(f"払戻: {payouts_file}")

    df = pd.read_csv(results_file, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['st']   = pd.to_numeric(df['st'],   errors='coerce')
    df['tenji_time'] = pd.to_numeric(df['tenji_time'], errors='coerce')
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
    df['wave']       = pd.to_numeric(df['wave'],       errors='coerce')
    df = df.sort_values(['date', 'jyo_cd', 'race_no', 'course']).reset_index(drop=True)
    print(f"  {len(df):,}行, 期間: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 払戻マージ
    if payouts_file and Path(payouts_file).exists():
        pdf = pd.read_csv(payouts_file, encoding='utf-8-sig')
        pdf['date'] = pd.to_datetime(pdf['date'])
        df = df.merge(pdf, on=['date', 'jyo_cd', 'race_no'], how='left')
        print(f"  払戻マージ済み. 2連単あり: {df['payout_exacta'].notna().sum():,}行")

    return df


def _rolling_rate(df, groupby_cols, value_col, window_days, min_periods=5, date_col='date'):
    """
    各行の過去window_days日間の value_col の平均を計算（当日は除く）。
    groupby_cols でグループ化。pandas time-based rolling で高速化。
    """
    df_work = df[groupby_cols + [date_col, value_col]].copy()
    df_work['__orig_idx__'] = df.index
    df_work = df_work.sort_values(date_col)

    window_str = f'{window_days}D'
    results = []

    for _, gdf in df_work.groupby(groupby_cols, sort=False):
        gdf = gdf.sort_values(date_col)
        s = gdf.set_index(date_col)[value_col]
        # closed='left': [t-window, t) — 当日を除く
        rolled = s.rolling(window_str, min_periods=min_periods, closed='left').mean()
        results.append(pd.Series(rolled.values, index=gdf['__orig_idx__'].values))

    return pd.concat(results).sort_index()


def _rolling_n(df, groupby_cols, value_col, n, date_col='date', min_periods=1):
    """
    各行の直近n走 (当日を除く) の value_col の平均。
    count-based window: shift(1).rolling(n)
    """
    df_work = df[groupby_cols + [date_col, value_col]].copy()
    df_work['__orig_idx__'] = df.index
    results = []
    for _, gdf in df_work.groupby(groupby_cols, sort=False):
        gdf = gdf.sort_values(date_col)
        rolled = gdf[value_col].shift(1).rolling(n, min_periods=min_periods).mean()
        results.append(pd.Series(rolled.values, index=gdf['__orig_idx__'].values))
    return pd.concat(results).sort_index()


def _compute_streaks(df, groupby_cols, win_col, date_col='date'):
    """
    直近連続勝利数 / 直近連続未勝利数 を行ごとに返す (当日を除く: shift済み)。
    Returns: (form_win_streak, form_no_win_streak) の Series ペア
    """
    df_work = df[groupby_cols + [date_col, win_col]].copy()
    df_work['__orig_idx__'] = df.index

    win_streaks  = np.zeros(len(df), dtype=np.int16)
    lose_streaks = np.zeros(len(df), dtype=np.int16)

    for _, gdf in df_work.groupby(groupby_cols, sort=False):
        gdf  = gdf.sort_values(date_col)
        idxs = gdf['__orig_idx__'].values
        wins = gdf[win_col].values.astype(int)
        n    = len(wins)
        ws   = np.zeros(n, dtype=np.int16)
        ls   = np.zeros(n, dtype=np.int16)
        # i 番の「直前まで」の連続数を構築
        for i in range(1, n):
            if wins[i - 1] == 1:
                ws[i] = ws[i - 1] + 1
                ls[i] = 0
            else:
                ws[i] = 0
                ls[i] = ls[i - 1] + 1
        win_streaks[idxs]  = ws
        lose_streaks[idxs] = ls

    return (pd.Series(win_streaks,  index=df.index),
            pd.Series(lose_streaks, index=df.index))


def add_recent_form_features(df, ns=(3, 5, 10)):
    """
    直近N走ベースのフォーム特徴量を追加する。

    生成特徴量 (走数ベース・当日除外):
      form_win_{n}          直近n走勝率
      form_top2_{n}         直近n走2着以内率
      form_avg_rank_{n}     直近n走平均着順
      form_avg_st_{n}       直近n走平均ST
      form_trend_win        win_5 - win_30  (短期上昇トレンド)
      form_trend_rank       avg_rank_3 - avg_rank_10 (着順改善度: 負が改善)
      form_win_streak       直近連続1着数
      form_no_win_streak    直近連続未勝利数
      form_venue_win_5      同場直近5走勝率
      form_venue_top2_5     同場直近5走2着以内率
    """
    print("直近フォーム特徴量を計算中...")
    df = df.copy()
    df['win']    = (df['rank'] == 1).astype(float)
    df['top2']   = (df['rank'] <= 2).astype(float)
    st_col = 'st_val' if 'st_val' in df.columns else 'st'

    # トレンド計算に必要な n を追加（未指定でも必ず生成）
    required = sorted(set(ns) | {3, 5, 10})
    for n in required:
        label = f"直近{n}走" + ("" if n in ns else " (トレンド用)")
        print(f"  {label}...")
        df[f'form_win_{n}']      = _rolling_n(df, ['racer_id'], 'win',  n)
        df[f'form_top2_{n}']     = _rolling_n(df, ['racer_id'], 'top2', n)
        df[f'form_avg_rank_{n}'] = _rolling_n(df, ['racer_id'], 'rank', n)
        df[f'form_avg_st_{n}']   = _rolling_n(df, ['racer_id'], st_col, n)

    # 短期トレンド (直近5走 vs 直近30走)
    win_30 = _rolling_n(df, ['racer_id'], 'win', 30)
    df['form_trend_win']  = df['form_win_5']      - win_30
    df['form_trend_rank'] = df['form_avg_rank_3'] - df['form_avg_rank_10']

    # 連勝・連敗ストリーク
    print("  連勝/連敗ストリーク...")
    df['form_win_streak'], df['form_no_win_streak'] = _compute_streaks(
        df, ['racer_id'], 'win'
    )

    # 同場直近成績
    print("  同場直近5走...")
    df['form_venue_win_5']  = _rolling_n(df, ['racer_id', 'jyo_cd'], 'win',  5)
    df['form_venue_top2_5'] = _rolling_n(df, ['racer_id', 'jyo_cd'], 'top2', 5)

    return df


def add_racer_features(df, windows=(90, 180)):
    """選手ごとのローリング統計"""
    print("選手特徴量を計算中...")
    df = df.copy()
    df['win']    = (df['rank'] == 1).astype(float)
    df['top2']   = (df['rank'] <= 2).astype(float)
    df['top3']   = (df['rank'] <= 3).astype(float)
    df['st_val'] = df['st'].clip(lower=-0.1, upper=0.5)

    for w in windows:
        print(f"  選手 × コース × {w}日窓...")
        for col, out in [('win', f'racer_course_win{w}'),
                         ('top2', f'racer_course_top2_{w}'),
                         ('st_val', f'racer_course_st{w}')]:
            df[out] = _rolling_rate(df, ['racer_id', 'course'], col, w)
        # コースに関係なく全体勝率
        for col, out in [('win', f'racer_win{w}'),
                         ('top2', f'racer_top2_{w}')]:
            df[out] = _rolling_rate(df, ['racer_id'], col, w)

    return df


def add_motor_features(df, window=60):
    """モーター相対性能 (会場内平均との比較)"""
    print("モーター特徴量を計算中...")
    df = df.copy()
    df['top2'] = (df['rank'] <= 2).astype(float) if 'top2' not in df.columns else df['top2']

    # モーター2連率
    df[f'motor_top2_{window}'] = _rolling_rate(df, ['jyo_cd', 'motor_no'], 'top2', window)
    # 会場平均
    df[f'venue_avg_top2_{window}'] = _rolling_rate(df, ['jyo_cd'], 'top2', window, min_periods=100)
    # 相対値
    df[f'motor_rel_{window}'] = df[f'motor_top2_{window}'] - df[f'venue_avg_top2_{window}']

    return df


def add_venue_course_features(df, window=180):
    """会場 × コースのバイアス"""
    print("会場・コース特徴量を計算中...")
    df = df.copy()
    df['win'] = (df['rank'] == 1).astype(float) if 'win' not in df.columns else df['win']

    df[f'venue_course_win{window}'] = _rolling_rate(
        df, ['jyo_cd', 'course'], 'win', window, min_periods=50
    )
    return df


def add_weather_features(df):
    """天候・風速・波の数値化"""
    df = df.copy()
    weather_map = {'晴': 1, '曇': 0, '雨': -1, '雪': -2}
    df['weather_num'] = df['weather'].map(weather_map).fillna(0)
    df['wind_speed'] = df['wind_speed'].fillna(0).clip(0, 15)
    df['wave'] = df['wave'].fillna(0).clip(0, 50)
    # 荒天スコア（大きいほど荒れ）
    df['rough_score'] = df['wind_speed'] * 0.5 + df['wave'] * 0.1 - df['weather_num']
    return df


def add_race_context(df):
    """レース内の相対情報（展示タイム最速など）"""
    df = df.copy()
    # 展示タイム: 小さいほど速い（良い）
    grp = df.groupby(['date', 'jyo_cd', 'race_no'])
    df['tenji_rank'] = grp['tenji_time'].rank(method='min')
    df['tenji_best']  = grp['tenji_time'].transform('min')
    df['tenji_rel']   = df['tenji_time'] - df['tenji_best']

    # 前走ST相対
    df['st_rel'] = df['st_val'] if 'st_val' in df.columns else df['st'].clip(-0.1, 0.5)
    df['st_rel'] = df['st_rel'] - grp['st'].transform('mean')

    # レースタイプ（本番度）
    race_type_map = {'優勝戦': 3, '準優': 2, '予選': 1, '一般': 1}
    df['race_importance'] = df['race_type'].map(race_type_map).fillna(1)

    return df


def add_implied_probability(df):
    """
    払戻から逆算した実績確率を追加。
    payout_win → implied_win_prob = 100 / payout_win × (1 - take_rate補正)

    実戦で使うには事前情報（オッズ）が必要だが、
    ここでは「過去の類似レースの平均払戻」を使った期待値の学習に活用。
    """
    df = df.copy()
    if 'payout_win' in df.columns:
        df['implied_win_prob'] = 100.0 / df['payout_win'].clip(lower=100)
    if 'payout_exacta' in df.columns:
        df['implied_exacta_prob'] = 100.0 / df['payout_exacta'].clip(lower=100)
    if 'payout_trifecta' in df.columns:
        df['implied_trifecta_prob'] = 100.0 / df['payout_trifecta'].clip(lower=100)
    return df


def build_race_level_features(df):
    """
    1行 = 1レース でのレース予測用特徴量を構築。pivot_table で高速化。
    実戦では「レース前に知れる情報」のみ使用。
    """
    race_keys = ['date', 'jyo_cd', 'race_no']

    feature_cols = [
        'racer_course_win90', 'racer_course_top2_90',
        'racer_course_win180', 'racer_course_top2_180',
        'racer_course_st90',
        'racer_win90', 'racer_top2_90',
        'motor_rel_60', 'motor_top2_60',
        'tenji_rel', 'st_rel',
        'venue_course_win180',
        # 直近フォーム
        'form_win_3',   'form_win_5',   'form_win_10',
        'form_top2_3',  'form_top2_5',  'form_top2_10',
        'form_avg_rank_3', 'form_avg_rank_5', 'form_avg_rank_10',
        'form_avg_st_5', 'form_avg_st_10',
        'form_trend_win', 'form_trend_rank',
        'form_win_streak', 'form_no_win_streak',
        'form_venue_win_5', 'form_venue_top2_5',
    ]
    available_feat = [c for c in feature_cols if c in df.columns]

    # コース別特徴量をピボット (各コースの列を c{course}_{feat} に展開)
    pivot = df.pivot_table(
        index=race_keys, columns='course', values=available_feat, aggfunc='first'
    )
    pivot.columns = [f'c{int(c)}_{f}' for f, c in pivot.columns]
    pivot = pivot.reset_index()

    # レース共通情報 (最初の行)
    common_cols = ['weather_num', 'wind_speed', 'wave', 'rough_score', 'race_importance']
    payout_cols = ['payout_win', 'payout_exacta', 'payout_trifecta', 'exacta_1st', 'exacta_2nd']
    all_common = [c for c in common_cols + payout_cols if c in df.columns]

    common = df.groupby(race_keys)[all_common].first().reset_index()

    # 1着コース
    winner = (df[df['rank'] == 1]
              .groupby(race_keys)['course']
              .first()
              .rename('winner_course')
              .reset_index())

    race_df = pivot.merge(common, on=race_keys, how='left')
    race_df = race_df.merge(winner, on=race_keys, how='left')

    return race_df


def build_features(results_file=None, payouts_file=None, out_prefix='v3'):
    """メイン処理"""
    df = load_data(results_file, payouts_file)

    df = add_weather_features(df)
    df = add_race_context(df)
    df = add_racer_features(df, windows=(90, 180))
    df = add_motor_features(df, window=60)
    df = add_venue_course_features(df, window=180)
    df = add_recent_form_features(df, ns=(3, 5, 10))
    df = add_implied_probability(df)

    print("レース単位特徴量を構築中...")
    race_df = build_race_level_features(df)
    race_df = race_df.dropna(subset=['winner_course'])

    # 時系列分割 (80/10/10)
    race_df = race_df.sort_values('date').reset_index(drop=True)
    n = len(race_df)
    i_val  = int(n * 0.80)
    i_test = int(n * 0.90)

    train = race_df.iloc[:i_val]
    val   = race_df.iloc[i_val:i_test]
    test  = race_df.iloc[i_test:]

    feat_cols = [c for c in race_df.columns if c.startswith('c') and '_' in c]
    feat_cols += ['weather_num', 'wind_speed', 'wave', 'rough_score', 'race_importance']

    print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    print(f"特徴量数: {len(feat_cols)}")

    train.to_csv(PROC_DIR / f'{out_prefix}_train.csv', index=False)
    val.to_csv(  PROC_DIR / f'{out_prefix}_val.csv',   index=False)
    test.to_csv( PROC_DIR / f'{out_prefix}_test.csv',  index=False)

    with open(PROC_DIR / f'{out_prefix}_feature_cols.txt', 'w') as f:
        f.write('\n'.join(feat_cols))

    print(f"保存完了: {PROC_DIR}/{out_prefix}_*.csv")
    return train, val, test, feat_cols


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=None)
    parser.add_argument('--payouts', default=None)
    parser.add_argument('--prefix',  default='v3')
    args = parser.parse_args()
    build_features(args.results, args.payouts, args.prefix)
