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
    groupby_cols でグループ化。
    """
    df = df.copy()
    key = f"{'_'.join(str(c) for c in groupby_cols)}_roll{window_days}_{value_col}"
    result = []

    for group_vals, gdf in df.groupby(groupby_cols, sort=False):
        gdf = gdf.sort_values(date_col)
        dates = gdf[date_col].values
        vals  = gdf[value_col].values
        out   = np.full(len(gdf), np.nan)
        for i in range(len(gdf)):
            cutoff = dates[i] - np.timedelta64(window_days, 'D')
            mask = (dates < dates[i]) & (dates >= cutoff)
            if mask.sum() >= min_periods:
                out[i] = np.nanmean(vals[mask])
        result.append(pd.Series(out, index=gdf.index))

    return pd.concat(result).sort_index()


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
    1行 = 1レース (コース1の選手視点) でのレース予測用特徴量を構築。
    実戦では「レース前に知れる情報」のみ使用。
    """
    race_keys = ['date', 'jyo_cd', 'race_no']

    # 各コースの選手情報をピボット
    courses = [1, 2, 3, 4, 5, 6]
    feature_cols = [
        'racer_course_win90', 'racer_course_top2_90',
        'racer_course_win180', 'racer_course_top2_180',
        'racer_course_st90',
        'racer_win90', 'racer_top2_90',
        'motor_rel_60', 'motor_top2_60',
        'tenji_rel', 'st_rel',
        'venue_course_win180',
    ]

    records = []
    for (date, jyo_cd, race_no), gdf in df.groupby(race_keys):
        row = {'date': date, 'jyo_cd': jyo_cd, 'race_no': race_no}
        # 会場・天候は共通
        row['weather_num']  = gdf['weather_num'].iloc[0] if 'weather_num' in gdf.columns else 0
        row['wind_speed']   = gdf['wind_speed'].iloc[0] if 'wind_speed' in gdf.columns else 0
        row['wave']         = gdf['wave'].iloc[0] if 'wave' in gdf.columns else 0
        row['rough_score']  = gdf['rough_score'].iloc[0] if 'rough_score' in gdf.columns else 0
        row['race_importance'] = gdf['race_importance'].iloc[0] if 'race_importance' in gdf.columns else 1

        # コース別の払戻（ターゲット）
        if 'payout_win' in gdf.columns:
            row['payout_win'] = gdf['payout_win'].iloc[0]
        if 'payout_exacta' in gdf.columns:
            row['payout_exacta'] = gdf['payout_exacta'].iloc[0]
            row['exacta_1st'] = gdf['exacta_1st'].iloc[0] if 'exacta_1st' in gdf.columns else np.nan
            row['exacta_2nd'] = gdf['exacta_2nd'].iloc[0] if 'exacta_2nd' in gdf.columns else np.nan
        if 'payout_trifecta' in gdf.columns:
            row['payout_trifecta'] = gdf['payout_trifecta'].iloc[0]

        # 1着コース
        winner = gdf[gdf['rank'] == 1]
        row['winner_course'] = int(winner['course'].iloc[0]) if len(winner) > 0 else np.nan

        for course in courses:
            cdf = gdf[gdf['course'] == course]
            if len(cdf) == 0:
                for fc in feature_cols:
                    row[f'c{course}_{fc}'] = np.nan
            else:
                r = cdf.iloc[0]
                for fc in feature_cols:
                    row[f'c{course}_{fc}'] = r.get(fc, np.nan)

        records.append(row)

    return pd.DataFrame(records)


def build_features(results_file=None, payouts_file=None, out_prefix='v3'):
    """メイン処理"""
    df = load_data(results_file, payouts_file)

    df = add_weather_features(df)
    df = add_race_context(df)
    df = add_racer_features(df, windows=(90, 180))
    df = add_motor_features(df, window=60)
    df = add_venue_course_features(df, window=180)
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
