#!/usr/bin/env python3
"""
download_full.py
2018年〜現在まで全データを並列ダウンロード。チェックポイント付き。
Usage:
    python download_full.py --start 2018-01-01 --end 2026-03-19 --workers 20
"""
import os, re, time, sys, json
import requests, lhafile, pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from download_real import parse_venue_section, RAW_DIR, BASE_URL

CHECKPOINT_FILE = os.path.join(RAW_DIR, 'download_full_checkpoint.json')
OUT_RESULTS = os.path.join(RAW_DIR, 'race_results_full.csv')
OUT_PAYOUTS = os.path.join(RAW_DIR, 'race_payouts_full.csv')


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f)['done'])
    return set()


def save_checkpoint(done_dates):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'done': sorted(done_dates)}, f)


def download_one(date):
    ym = date.strftime('%Y%m')
    ymd = date.strftime('%y%m%d')
    url = BASE_URL.format(ym=ym, ymd=ymd)
    date_str = date.strftime('%Y-%m-%d')
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=40)
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
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return date_str, [], []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default='2026-03-19')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--chunk', type=int, default=500, help='何日ごとにCSVに書き出すか')
    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d')
    end   = datetime.strptime(args.end,   '%Y-%m-%d')

    all_dates = []
    cur = start
    while cur <= end:
        all_dates.append(cur)
        cur += timedelta(days=1)

    done_dates = load_checkpoint()
    pending = [d for d in all_dates if d.strftime('%Y-%m-%d') not in done_dates]

    print(f"期間: {args.start} ~ {args.end} ({len(all_dates)}日)")
    print(f"済み: {len(done_dates)}日, 残り: {len(pending)}日")
    print(f"並列数: {args.workers}, チャンクサイズ: {args.chunk}日")

    # 既存ファイルがあれば追記モード
    results_exists = os.path.exists(OUT_RESULTS)
    payouts_exists = os.path.exists(OUT_PAYOUTS)

    all_records, all_payouts = [], []
    processed = 0

    def flush(records, payouts, force=False):
        nonlocal results_exists, payouts_exists
        if records:
            df = pd.DataFrame(records)
            col_order = ['date','jyo_cd','jyo_name','race_no','course',
                         'racer_id','motor_no','boat_no',
                         'st','tenji_time','race_time','rank',
                         'kimarite','race_type','weather','wind_speed','wave','boat']
            cols = [c for c in col_order if c in df.columns]
            df = df[cols]
            df.to_csv(OUT_RESULTS, mode='a', header=not results_exists, index=False, encoding='utf-8-sig')
            results_exists = True
        if payouts:
            pdf = pd.DataFrame(payouts)
            pdf.to_csv(OUT_PAYOUTS, mode='a', header=not payouts_exists, index=False, encoding='utf-8-sig')
            payouts_exists = True

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, d): d for d in pending}
        chunk_records, chunk_payouts, chunk_done = [], [], []

        for future in as_completed(futures):
            date_str, records, payouts = future.result()
            chunk_records.extend(records)
            chunk_payouts.extend(payouts)
            chunk_done.append(date_str)
            processed += 1

            if processed % 30 == 0 or processed == len(pending):
                total_rows = len(done_dates) * 0 + processed  # approximate
                print(f"  [{processed}/{len(pending)}] 今回 {len(chunk_records):,}行 バッファ中")

            if len(chunk_done) >= args.chunk or processed == len(pending):
                flush(chunk_records, chunk_payouts)
                done_dates.update(chunk_done)
                save_checkpoint(done_dates)
                print(f"  → チャンク保存: {len(chunk_done)}日分, 累計済み: {len(done_dates)}日")
                chunk_records, chunk_payouts, chunk_done = [], [], []

    # 最終確認
    if os.path.exists(OUT_RESULTS):
        df = pd.read_csv(OUT_RESULTS)
        df = df.sort_values(['date','jyo_cd','race_no','course']).drop_duplicates()
        df.to_csv(OUT_RESULTS, index=False, encoding='utf-8-sig')
        print(f"\n完了! 結果: {len(df):,}行 → {OUT_RESULTS}")
        print(f"  期間: {df['date'].min()} ~ {df['date'].max()}")

    if os.path.exists(OUT_PAYOUTS):
        pdf = pd.read_csv(OUT_PAYOUTS)
        pdf = pdf.drop_duplicates()
        pdf.to_csv(OUT_PAYOUTS, index=False, encoding='utf-8-sig')
        print(f"払戻: {len(pdf):,}レース → {OUT_PAYOUTS}")


if __name__ == '__main__':
    main()
