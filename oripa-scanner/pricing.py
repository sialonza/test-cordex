"""
実勢価格ルックアップ

メルカリ/ヤフオク/カードラッシュ等からカードの実勢価格を取得。
サイトが提示する「価値」は往々にして水増しされているため、必ず実勢価格と照合する。

戦略:
  1. キャッシュファーストで価格取得（同日の価格は再利用）
  2. メルカリの「売り切れ」相場（過去3ヶ月の直近10件の中央値）が最も現実的
  3. Yahoo!オークションの落札価格もクロスチェック用に取得
"""
import json
import os
import urllib.request
import urllib.parse
import re
import time
from datetime import datetime, timedelta

CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "price_cache.json")
CACHE_TTL_HOURS = 24


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _is_fresh(timestamp_iso: str) -> bool:
    try:
        t = datetime.fromisoformat(timestamp_iso)
        return datetime.utcnow() - t < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


def get_mercari_median_price(card_name: str, max_items: int = 20) -> float:
    """
    メルカリで「売り切れ」状態の直近N件から中央値を取得。

    URL: https://jp.mercari.com/search?keyword={query}&status=sold_out
    レスポンスは HTML + 埋め込みJSON or GraphQL API
    """
    cache = _load_cache()
    cache_key = f"mercari:{card_name}"

    if cache_key in cache and _is_fresh(cache[cache_key].get("ts", "")):
        return cache[cache_key]["price"]

    try:
        query = urllib.parse.quote(card_name)
        url = f"https://jp.mercari.com/search?keyword={query}&status=sold_out&order=desc&sort=created_time"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept-Language": "ja,en;q=0.9",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # メルカリは __NEXT_DATA__ に JSON が埋め込まれる
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        prices = []
        if m:
            try:
                data = json.loads(m.group(1))
                # items パスは変わる可能性あり（要メンテ）
                items = (
                    data.get("props", {})
                    .get("pageProps", {})
                    .get("initialState", {})
                    .get("search", {})
                    .get("items", [])
                )
                for item in items[:max_items]:
                    p = item.get("price") or item.get("priceInYen")
                    if p and p > 0:
                        prices.append(int(p))
            except Exception:
                pass

        # フォールバック: 正規表現で価格を抜き出す
        if not prices:
            for m2 in re.finditer(r'"price":(\d+)', html):
                p = int(m2.group(1))
                if 100 <= p <= 5000000:  # 妥当な範囲
                    prices.append(p)
                if len(prices) >= max_items:
                    break

        if not prices:
            return 0.0

        prices.sort()
        median = prices[len(prices) // 2]

        cache[cache_key] = {"price": float(median), "ts": datetime.utcnow().isoformat(),
                            "n": len(prices)}
        _save_cache(cache)
        time.sleep(1.0)  # rate limit
        return float(median)
    except Exception as e:
        print(f"  [WARN] mercari lookup failed for '{card_name}': {e}")
        return 0.0


def enrich_prizes_with_market_prices(listing):
    """OripaListing の各 prize に market_value を埋める"""
    for prize in listing.prizes:
        if prize.market_value is None or prize.market_value == 0:
            prize.market_value = get_mercari_median_price(prize.name)
    return listing
