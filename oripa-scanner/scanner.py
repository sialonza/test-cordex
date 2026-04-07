#!/usr/bin/env python3
"""
オリパ期待値スキャナー メインCLI

使い方:
  python3 scanner.py --once                  # 1回スキャン
  python3 scanner.py --watch --interval 300  # 5分間隔で継続監視
  python3 scanner.py --mock data/sample.json # モックデータでテスト
  python3 scanner.py --threshold 1.0         # RTP 100%以上でアラート（デフォルトは1.0 = +EV）
  python3 scanner.py --no-prices             # 実勢価格ルックアップをスキップ（サイト提示値で計算）
"""
import argparse
import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scrapers import DopaScraper
from ev_engine import compute_ev, format_ev_report
from notify import notify_all, notify_console
from pricing import enrich_prizes_with_market_prices
from trust import effective_ev, record_purchase, site_stats


def get_scrapers(mock_file: str = None, har_file: str = None,
                 cookie: str = None):
    """有効なスクレイパーのリストを返す"""
    return [
        DopaScraper(mock_file=mock_file, har_file=har_file, cookie=cookie),
        # TODO: CloveScraper, IrisScraper, STOCKSScraper...
    ]


def scan_once(scrapers, use_market_prices: bool = True, threshold: float = 1.0,
              verbose: bool = True):
    """
    1回スキャン実行

    threshold: RTPがこの値以上ならアラート（1.0 = 100% = +EV）
    """
    all_results = []

    for scraper in scrapers:
        try:
            listings = scraper.fetch_pools()
        except Exception as e:
            print(f"[ERR] {scraper.SITE_NAME}: {e}")
            continue

        if not listings:
            if verbose:
                print(f"[{scraper.SITE_NAME}] 0 listings")
            continue

        print(f"[{scraper.SITE_NAME}] {len(listings)} listings fetched")

        for listing in listings:
            # 実勢価格で補強
            if use_market_prices:
                listing = enrich_prizes_with_market_prices(listing)

            ev = compute_ev(listing, use_market_value=use_market_prices)

            # 信頼係数で割り引いた実効EVを重ね掛け
            eff = effective_ev(listing.site, ev["ev"], ev["rtp"])
            ev["effective_ev"] = eff["effective_ev"]
            ev["effective_rtp"] = eff["effective_rtp"]
            ev["trust"] = eff["trust"]
            ev["flags"] = eff["flags"]

            all_results.append((listing, ev))

            if verbose:
                print("  " + format_ev_report(listing, ev).replace("\n", "\n  "))
                if eff["flags"]:
                    for f in eff["flags"]:
                        print(f"     {f}")
                print(f"     trust={eff['trust']} → 実効RTP {eff['effective_rtp']*100:.1f}%")

            # アラート判定: 実効RTPが閾値超え かつ 赤旗なし
            if eff["effective_rtp"] >= threshold and not eff["flags"]:
                alert_msg = (
                    f"+EV オリパ検出！\n"
                    f"{format_ev_report(listing, ev)}\n"
                    f"URL: {listing.url or 'N/A'}\n"
                    f"今すぐ購入検討"
                )
                notify_all(alert_msg)

    # RTPでソート
    all_results.sort(key=lambda x: -x[1]["rtp"])
    return all_results


def print_summary(results):
    """スキャン結果のサマリーを表示"""
    if not results:
        print("\n結果なし")
        return

    print(f"\n{'='*70}")
    print(f"  スキャン結果サマリー ({len(results)}件)")
    print(f"{'='*70}")
    print(f"  {'Site':<8} {'Name':<30} {'Price':>8} {'EV':>8} {'RTP':>7}  ")
    print(f"  {'-'*70}")
    for listing, ev in results[:20]:
        name = listing.name[:28]
        mark = "+" if ev["is_positive"] else " "
        print(f"  {listing.site:<8} {name:<30} ¥{listing.pack_price:>6,.0f} "
              f"¥{ev['ev']:>6,.0f} {ev['rtp']*100:>5.1f}% {mark}")

    pos_count = sum(1 for _, ev in results if ev["is_positive"])
    avg_rtp = sum(ev["rtp"] for _, ev in results) / len(results)
    print(f"\n  +EV: {pos_count}/{len(results)}  平均RTP: {avg_rtp*100:.1f}%")


def inspect(oripa_id: str, mock_file: str = None):
    """特定オリパの詳細EV分析"""
    for scraper in get_scrapers(mock_file):
        listings = scraper.fetch_pools()
        for listing in listings:
            if listing.oripa_id == oripa_id:
                listing = enrich_prizes_with_market_prices(listing)
                ev = compute_ev(listing)
                print(format_ev_report(listing, ev))
                print(f"\n  === プール詳細 ===")
                print(f"  {'カード名':<30} {'残/総':>8} {'提示値':>10} {'実勢':>10}")
                for p in sorted(listing.prizes, key=lambda x: -(x.market_value or x.stated_value)):
                    mv = f"¥{p.market_value:,.0f}" if p.market_value else "N/A"
                    print(f"  {p.name[:28]:<30} {p.count_remaining}/{p.count_total:>5} "
                          f"¥{p.stated_value:>8,.0f} {mv:>10}")
                return
    print(f"Oripa not found: {oripa_id}")


def main():
    parser = argparse.ArgumentParser(description="オリパ期待値スキャナー")
    parser.add_argument("--once", action="store_true", help="1回だけスキャン")
    parser.add_argument("--watch", action="store_true", help="継続監視")
    parser.add_argument("--interval", type=int, default=300, help="監視間隔（秒）")
    parser.add_argument("--mock", type=str, help="モックJSONファイル")
    parser.add_argument("--har", type=str, help="DevToolsエクスポートHARファイル")
    parser.add_argument("--cookie", type=str,
                        help="DOPAセッションCookie（省略時は env DOPA_COOKIE）")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="アラート閾値（RTP、1.0=100%%）")
    parser.add_argument("--no-prices", action="store_true",
                        help="実勢価格ルックアップをスキップ")
    parser.add_argument("--inspect", type=str, help="特定オリパIDの詳細表示")
    parser.add_argument("--quiet", action="store_true", help="詳細出力を抑制")
    parser.add_argument("--trust", type=str, metavar="SITE",
                        help="サイトの信頼度統計を表示")
    parser.add_argument("--record", nargs=6,
                        metavar=("SITE", "ID", "NAME", "PRICE", "PACKS", "ACTUAL"),
                        help="購入結果を記録: site id name pack_price packs actual_return")

    args = parser.parse_args()

    if args.trust:
        import pprint
        pprint.pprint(site_stats(args.trust))
        return

    if args.record:
        site, oid, name, price, packs, actual = args.record
        # 計算EVは現在のスキャン結果から拾う
        scrapers_tmp = get_scrapers(mock_file=args.mock, har_file=args.har,
                                    cookie=args.cookie)
        calc_ev_per_pack = 0.0
        for s in scrapers_tmp:
            for l in s.fetch_pools():
                if l.oripa_id == oid:
                    calc_ev_per_pack = compute_ev(l)["ev"]
        record_purchase(site, oid, name, float(price), int(packs),
                        calc_ev_per_pack, float(actual))
        print(f"recorded: {site}/{oid} calc_ev={calc_ev_per_pack:.0f} "
              f"actual={float(actual):.0f}")
        return

    if args.inspect:
        inspect(args.inspect, mock_file=args.mock)
        return

    scrapers = get_scrapers(mock_file=args.mock, har_file=args.har,
                            cookie=args.cookie)
    use_prices = not args.no_prices
    verbose = not args.quiet

    if args.watch:
        print(f"[WATCH] 継続監視開始 (interval={args.interval}s, threshold={args.threshold})")
        while True:
            try:
                results = scan_once(scrapers, use_market_prices=use_prices,
                                   threshold=args.threshold, verbose=verbose)
                print_summary(results)
                print(f"\n次回スキャン: {args.interval}秒後")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n[STOP] ユーザー中断")
                break
            except Exception as e:
                print(f"[ERR] {e}")
                time.sleep(args.interval)
    else:
        results = scan_once(scrapers, use_market_prices=use_prices,
                           threshold=args.threshold, verbose=verbose)
        print_summary(results)


if __name__ == "__main__":
    main()
