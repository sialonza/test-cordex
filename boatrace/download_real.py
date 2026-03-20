#!/usr/bin/env python3
"""
download_real.py
ボートレース公式サイトから実データをダウンロードし、CSVに変換する。

データソース: https://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh
1ファイルに全会場の1日分の成績データが含まれる。

Usage:
    python download_real.py --start 2025-01-01 --end 2025-01-31
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
    """
    レースヘッダー行から情報を抽出。
    例: '   1R       予選　　　　                 H1800m  晴　  風  北西　 2m  波　  1cm'
    """
    race_no = None
    race_type = ""
    weather = None
    wind_speed = None
    wave = None
    kimarite = ""

    m = re.search(r'(\d+)R', line)
    if m:
        race_no = int(m.group(1))

    # レース種別: 位置11-17付近
    race_type = line[11:17].strip().replace('\u3000', '').strip()

    # 天候
    for w in ['晴', '曇', '雨', '雪']:
        if w in line:
            weather = '曇り' if w == '曇' else w
            break

    # 風速: "風  北西　 2m" → 数字+m
    m = re.search(r'風\s*\S+\s*(\d+)m', line)
    if m:
        wind_speed = int(m.group(1))

    # 波高: "波　  1cm"
    m = re.search(r'波\s*(\d+)cm', line)
    if m:
        wave = int(m.group(1))

    # 決まり手: ヘッダー行末尾 ("逃げ", "差し" など)
    tail = line.rstrip()
    if len(tail) > 6:
        kimarite = tail[-6:].replace('\u3000', '').strip()

    return race_no, race_type, weather, wind_speed, wave, kimarite


def parse_result_line(line):
    """
    レース結果行から情報を抽出。
    例: '  01  1 3527 中　嶋　　誠一郎 57   59  6.87   1    0.13     1.48.5'
    ポジション (Unicode文字位置):
      [2:4]  着順  [6]    艇番  [8:12]  登番
      [22:24] モーター [27:29] ボート [31:36] 展示タイム
      [38]   進入コース [43:47] STタイミング [52:58] レースタイム
    """
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
        # レースタイム: "1.48.5" などの形式、着外は ".  . "
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


def parse_venue_section(lines, date_str, jyo_cd):
    """会場セクション (XXKBGNからXXKENDまで) をパース"""
    jyo_name = JYO_NAMES.get(jyo_cd, f"不明({jyo_cd})")
    records = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # レースヘッダー行の検出: "   1R       予選..." (数字R + H + m が含まれる)
        if re.search(r'\d+R', line) and 'H' in line and re.search(r'\dm', line):
            race_no, race_type, weather, wind_speed, wave, kimarite = parse_race_header(line)

            if race_no is None:
                i += 1
                continue

            # カラムヘッダー行 (着 艇 登番 ... ﾚｰｽﾀｲﾑ 逃げ) から決まり手を取得
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
                # 結果行: "  01  1 3527..."
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
            continue

        i += 1

    return records


def download_day(date):
    """1日分のレース結果をダウンロード・パースして返す"""
    ym = date.strftime('%Y%m')
    ymd = date.strftime('%y%m%d')
    url = BASE_URL.format(ym=ym, ymd=ymd)
    date_str = date.strftime('%Y-%m-%d')

    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []

        lha = lhafile.Lhafile(BytesIO(r.content))
        info = lha.infolist()
        if not info:
            return []

        content = lha.read(info[0].filename)
        text = content.decode('cp932', errors='replace')
        lines = text.splitlines()

        all_records = []

        # 会場セクション (XXKBGNからXXKENDまで) を抽出してパース
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
                records = parse_venue_section(section_lines, date_str, jyo_cd)
                all_records.extend(records)
            i += 1

        return all_records

    except Exception as e:
        print(f'  エラー ({date_str}): {e}')
        return []


def main():
    parser = argparse.ArgumentParser(description='ボートレース実データダウンロード')
    parser.add_argument('--start', default='2025-01-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-01-31', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--out', default='race_results_real.csv', help='出力ファイル名')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    print('=' * 60)
    print('ボートレース実データ ダウンロード (cstenmt/boatrace ベース)')
    print('=' * 60)
    print(f'対象期間: {args.start} ~ {args.end}')
    print(f'データソース: {BASE_URL}')
    print()

    all_records = []
    current = start_date

    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        print(f'  {date_str} ... ', end='', flush=True)
        records = download_day(current)
        print(f'{len(records)} 行')
        all_records.extend(records)
        time.sleep(0.5)
        current += timedelta(days=1)

    if not all_records:
        print('\nデータが取得できませんでした。')
        return

    df = pd.DataFrame(all_records)

    # カラム順を整理
    col_order = [
        'date', 'jyo_cd', 'jyo_name', 'race_no', 'course',
        'racer_id', 'motor_no', 'boat_no',
        'st', 'tenji_time', 'race_time', 'rank',
        'kimarite', 'race_type', 'weather', 'wind_speed', 'wave',
        'boat',
    ]
    cols = [c for c in col_order if c in df.columns]
    df = df[cols]

    out_path = os.path.join(RAW_DIR, args.out)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print()
    print(f'合計レコード数: {len(df):,} 行')
    print(f'会場数: {df["jyo_cd"].nunique()}')
    print(f'日数: {df["date"].nunique()}')
    print(f'保存先: {out_path}')
    print()
    print('サンプル (先頭5行):')
    print(df.head().to_string())
    print('\n完了!')


if __name__ == '__main__':
    main()
