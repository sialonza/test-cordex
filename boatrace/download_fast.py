#!/usr/bin/env python3
"""
download_fast.py
並列ダウンロードで高速にデータ取得。ThreadPoolExecutor使用。
"""
import os, re, time, sys
import requests, lhafile, pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from download_real import (
    parse_venue_section, RAW_DIR, BASE_URL, JYO_NAMES
)

def download_one(date):
    ym = date.strftime('%Y%m')
    ymd = date.strftime('%y%m%d')
    url = BASE_URL.format(ym=ym, ymd=ymd)
    date_str = date.strftime('%Y-%m-%d')
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return date_str, [], []
            lha = lhafile.Lhafile(BytesIO(r.content))
            info = lha.infolist()
            if not info:
                return date_str, [], []
            content = lha.read(info[0].filename)
            text = content.decode('cp932', errors='replace')
            lines = text.splitlines()
            all_records, all_payouts = [], []
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
            return date_str, all_records, all_payouts
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return date_str, [], []

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2025-07-01')
    parser.add_argument('--end', default='2026-03-17')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur += timedelta(days=1)

    print(f"期間: {args.start} ~ {args.end} ({len(dates)}日)")
    print(f"並列数: {args.workers}")

    all_records, all_payouts = [], []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, d): d for d in dates}
        for future in as_completed(futures):
            date_str, records, payouts = future.result()
            all_records.extend(records)
            all_payouts.extend(payouts)
            done += 1
            if done % 30 == 0 or done <= 3 or done == len(dates):
                print(f"  [{done}/{len(dates)}] 累計 {len(all_records):,} 行, {len(all_payouts):,} レース")

    if all_records:
        df = pd.DataFrame(all_records)
        col_order = [
            'date', 'jyo_cd', 'jyo_name', 'race_no', 'course',
            'racer_id', 'motor_no', 'boat_no',
            'st', 'tenji_time', 'race_time', 'rank',
            'kimarite', 'race_type', 'weather', 'wind_speed', 'wave', 'boat',
        ]
        cols = [c for c in col_order if c in df.columns]
        df = df[cols].sort_values(['date', 'jyo_cd', 'race_no', 'course']).reset_index(drop=True)
        out = os.path.join(RAW_DIR, 'race_results_real.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"\n結果: {len(df):,} 行 → {out}")
        print(f"  期間: {df['date'].min()} ~ {df['date'].max()}")

    if all_payouts:
        pdf = pd.DataFrame(all_payouts)
        pout = os.path.join(RAW_DIR, 'race_payouts_real.csv')
        pdf.to_csv(pout, index=False, encoding='utf-8-sig')
        print(f"払戻: {len(pdf):,} レース → {pout}")
        if 'payout_exacta' in pdf.columns:
            print(f"  2連単平均: {pdf['payout_exacta'].mean():,.0f}円")

    print("完了!")

if __name__ == '__main__':
    main()
