#!/usr/bin/env python3
"""
download_real.py
ボートレース公式サイトから実データをダウンロードし、CSVに変換する。

データソース: https://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh
1ファイルに全会場の1日分の成績データ＋払戻データが含まれる。

出力:
  - race_results_real.csv  : レース結果 (1行 = 1レーサー × 1レース)
  - race_payouts_real.csv  : 払戻金 (1行 = 1レース)

Usage:
    python download_real.py --start 2024-10-01 --end 2026-03-17
"""

import os
import re
import time
import argparse
import requests
import lhafile
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://www1.mbrace.or.jp/od2/K/{ym}/k{ymd}.lzh"

JYO_NAMES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島",
    "05": "多摩川", "06": "浜名湖", "07": "蒲郡", "08": "常滑",
    "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島",
    "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}


def parse_race_header(line):
    """レースヘッダー行から情報を抽出"""
    race_no = None
    race_type = ""
    weather = None
    wind_speed = None
    wave = None

    m = re.search(r'(\d+)R', line)
    if m:
        race_no = int(m.group(1))

    race_type = line[11:17].strip().replace('\u3000', '').strip()

    for w in ['晴', '曇', '雨', '雪']:
        if w in line:
            weather = '曇り' if w == '曇' else w
            break

    m = re.search(r'風\s*\S+\s*(\d+)m', line)
    if m:
        wind_speed = int(m.group(1))

    m = re.search(r'波\s*(\d+)cm', line)
    if m:
        wave = int(m.group(1))

    return race_no, race_type, weather, wind_speed, wave


def parse_result_line(line):
    """レース結果行から情報を抽出"""
    try:
        rank_str = line[2:4].strip()
        boat_str = line[6].strip()
        racer_id_str = line[8:12].strip()

        if not rank_str.isdigit() or not racer_id_str.isdigit():
            return None

        rank = int(rank_str)
        boat = int(boat_str) if boat_str.isdigit() else None
        racer_id = int(racer_id_str)

        motor_str = line[22:24].strip()
        boat_no_str = line[27:29].strip()
        tenji_str = line[31:36].strip()
        entry_str = line[38].strip() if len(line) >= 39 else ""
        st_str = line[43:47].strip() if len(line) >= 47 else ""
        race_time_str = line[52:58].strip() if len(line) >= 52 else ""

        motor_no = int(motor_str) if motor_str.isdigit() else None
        boat_no = int(boat_no_str) if boat_no_str.isdigit() else None
        tenji_time = float(tenji_str) if re.match(r'^\d+\.\d+$', tenji_str) else None
        course = int(entry_str) if entry_str.isdigit() else boat
        st = float(st_str) if re.match(r'^\d+\.\d+$', st_str) else None
        race_time = race_time_str if re.match(r'^\d+\.\d+\.\d+', race_time_str) else None

        return {
            'rank': rank,
            'boat': boat,
            'racer_id': racer_id,
            'motor_no': motor_no,
            'boat_no': boat_no,
            'tenji_time': tenji_time,
            'course': course,
            'st': st,
            'race_time': race_time,
        }
    except (ValueError, IndexError):
        return None


def parse_payouts(lines_after_results):
    """
    レース結果行の後にある払戻データを抽出。

    例:
        ２連単   1-2        250  人気     1
        ３連単   1-2-4      910  人気     2
    """
    payouts = {}
    for line in lines_after_results:
        # 2連単
        m = re.search(r'２連単\s+([\d]+-[\d]+)\s+(\d+)', line)
        if m:
            combo = m.group(1)
            parts = combo.split('-')
            payouts['exacta_1st'] = int(parts[0])
            payouts['exacta_2nd'] = int(parts[1])
            payouts['payout_exacta'] = int(m.group(2))

        # 2連複
        m = re.search(r'２連複\s+([\d]+-[\d]+)\s+(\d+)', line)
        if m:
            payouts['payout_quinella'] = int(m.group(2))

        # 3連単
        m = re.search(r'３連単\s+([\d]+-[\d]+-[\d]+)\s+(\d+)', line)
        if m:
            combo = m.group(1)
            parts = combo.split('-')
            payouts['trifecta_1st'] = int(parts[0])
            payouts['trifecta_2nd'] = int(parts[1])
            payouts['trifecta_3rd'] = int(parts[2])
            payouts['payout_trifecta'] = int(m.group(2))

        # 3連複
        m = re.search(r'３連複\s+([\d]+-[\d]+-[\d]+)\s+(\d+)', line)
        if m:
            payouts['payout_trio'] = int(m.group(2))

        # 単勝
        m = re.search(r'単勝\s+(\d+)\s+(\d+)', line)
        if m:
            payouts['payout_win'] = int(m.group(2))

    return payouts


def parse_venue_section(lines, date_str, jyo_cd):
    """会場セクション (XXKBGNからXXKENDまで) をパース"""
    jyo_name = JYO_NAMES.get(jyo_cd, f"不明({jyo_cd})")
    records = []
    payout_records = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # レースヘッダー行の検出
        if re.search(r'\d+R', line) and 'H' in line and re.search(r'\dm', line):
            race_no, race_type, weather, wind_speed, wave = parse_race_header(line)

            if race_no is None:
                i += 1
                continue

            # カラムヘッダー行から決まり手を取得
            kimarite = ""
            i += 1
            while i < len(lines) and not lines[i].startswith('---'):
                col_header = lines[i]
                if 'ﾚｰｽﾀｲﾑ' in col_header:
                    tail = col_header.rstrip()
                    kimarite = re.split(r'ﾚｰｽﾀｲﾑ', tail)[-1].replace('\u3000', '').strip()
                i += 1
            i += 1  # skip "---" line

            # 結果行を処理
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    break
                if re.match(r'\s{2}\d{2}\s+\d', line):
                    row = parse_result_line(line)
                    if row:
                        row.update({
                            'date': date_str,
                            'jyo_cd': jyo_cd,
                            'jyo_name': jyo_name,
                            'race_no': race_no,
                            'race_type': race_type,
                            'weather': weather,
                            'wind_speed': wind_speed,
                            'wave': wave,
                            'kimarite': kimarite if row['rank'] == 1 else '',
                        })
                        records.append(row)
                i += 1

            # 払戻データを処理 (結果行の後に続く)
            payout_lines = []
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    # 空行が2つ連続したら次のレースへ
                    if i + 1 < len(lines) and not lines[i + 1].strip():
                        break
                    i += 1
                    continue
                # 次のレースヘッダーが来たら終了
                if re.search(r'\d+R', line) and 'H' in line:
                    break
                payout_lines.append(line)
                i += 1

            payouts = parse_payouts(payout_lines)
            if payouts:
                payouts.update({
                    'date': date_str,
                    'jyo_cd': jyo_cd,
                    'race_no': race_no,
                })
                payout_records.append(payouts)

            continue

        i += 1

    return records, payout_records


def download_day(date, max_retries=3):
    """1日分のレース結果＋払戻をダウンロード・パースして返す"""
    ym = date.strftime('%Y%m')
    ymd = date.strftime('%y%m%d')
    url = BASE_URL.format(ym=ym, ymd=ymd)
    date_str = date.strftime('%Y-%m-%d')

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                return [], []

            lha = lhafile.Lhafile(BytesIO(r.content))
            info = lha.infolist()
            if not info:
                return [], []

            content = lha.read(info[0].filename)
            text = content.decode('cp932', errors='replace')
            lines = text.splitlines()

            all_records = []
            all_payouts = []

            i = 0
            while i < len(lines):
                m = re.match(r'(\d{2})KBGN', lines[i])
                if m:
                    jyo_cd = m.group(1)
                    end_marker = f'{jyo_cd}KEND'
                    section_lines = []
                    i += 1
                    while i < len(lines) and lines[i] != end_marker:
                        section_lines.append(lines[i])
                        i += 1
                    records, payouts = parse_venue_section(section_lines, date_str, jyo_cd)
                    all_records.extend(records)
                    all_payouts.extend(payouts)
                i += 1

            return all_records, all_payouts

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return [], []
        except Exception as e:
            print(f'  エラー ({date_str}): {e}')
            return [], []


def main():
    parser = argparse.ArgumentParser(description='ボートレース実データダウンロード')
    parser.add_argument('--start', default='2024-10-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2026-03-17', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--sleep', type=float, default=0.3, help='リクエスト間隔 (秒)')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    print('=' * 60)
    print('ボートレース実データ ダウンロード v2')
    print('=' * 60)
    print(f'対象期間: {args.start} ~ {args.end}')
    n_days = (end_date - start_date).days + 1
    print(f'日数: {n_days}日 (推定所要時間: {n_days * args.sleep / 60:.1f}分)')
    print()

    all_records = []
    all_payouts = []
    current = start_date
    day_count = 0

    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        day_count += 1
        if day_count % 30 == 1 or day_count <= 3:
            print(f'  [{day_count}/{n_days}] {date_str} ... ', end='', flush=True)
            records, payouts = download_day(current)
            print(f'{len(records)} 行, {len(payouts)} レース')
        else:
            records, payouts = download_day(current)
            if day_count % 30 == 0:
                print(f'  [{day_count}/{n_days}] {date_str} ... 累計 {len(all_records) + len(records):,} 行')
        all_records.extend(records)
        all_payouts.extend(payouts)
        time.sleep(args.sleep)
        current += timedelta(days=1)

    # --- 結果データ保存 ---
    if all_records:
        df = pd.DataFrame(all_records)
        col_order = [
            'date', 'jyo_cd', 'jyo_name', 'race_no', 'course',
            'racer_id', 'motor_no', 'boat_no',
            'st', 'tenji_time', 'race_time', 'rank',
            'kimarite', 'race_type', 'weather', 'wind_speed', 'wave', 'boat',
        ]
        cols = [c for c in col_order if c in df.columns]
        df = df[cols]
        out_path = os.path.join(RAW_DIR, 'race_results_real.csv')
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'\n結果データ: {len(df):,} 行 → {out_path}')
        print(f'  期間: {df["date"].min()} ~ {df["date"].max()}')
        print(f'  会場数: {df["jyo_cd"].nunique()}')
    else:
        print('\n結果データが取得できませんでした。')

    # --- 払戻データ保存 ---
    if all_payouts:
        pdf = pd.DataFrame(all_payouts)
        payout_path = os.path.join(RAW_DIR, 'race_payouts_real.csv')
        pdf.to_csv(payout_path, index=False, encoding='utf-8-sig')
        print(f'払戻データ: {len(pdf):,} レース → {payout_path}')
        if 'payout_exacta' in pdf.columns:
            print(f'  2連単平均配当: {pdf["payout_exacta"].mean():,.0f}円')
            print(f'  2連単中央値:   {pdf["payout_exacta"].median():,.0f}円')
    else:
        print('払戻データが取得できませんでした。')

    print('\n完了!')


if __name__ == '__main__':
    main()
