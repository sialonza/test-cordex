"""
DOPA スクレイパー

DOPA (dopa-game.jp) は Next.js App Router + RSC で構築されており、
  - トップHTMLは `noindex,nofollow`
  - コンテンツは認証済みサーバコンポーネント経由
  - 匿名での公開REST APIは存在しない (/api/* はすべて 404)

よって「匿名ブラインドスクレイピング」は実装不能。実運用では以下の
3方式をサポートする:

  1. mock_file   : 開発用JSON
  2. har_file    : ユーザが自分のブラウザセッションでDevTools→HARをエクスポート
                   → JSONレスポンスを解析 (ToS的にもクリーン、自分のセッション)
  3. live (cookie): 環境変数 DOPA_COOKIE に Cookie ヘッダをコピペ
                   → 候補エンドポイントに対して認証済みリクエスト

方式2 (HAR) が最も確実かつ安全。
"""
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
from .base import OripaSiteScraper, OripaListing, PrizeItem


# DOPA JSONレスポンスと共通モデルのフィールドマッピング。
# HARから拾った実レスポンスを見ながら調整する。
PRIZE_KEYS = {
    "name":            ["name", "title", "item_name", "card_name"],
    "count_total":     ["total", "count_total", "stock_total", "quantity"],
    "count_remaining": ["remaining", "count_remaining", "stock", "remain"],
    "stated_value":    ["value", "price", "stated_value", "market_price"],
    "rarity":          ["rarity", "rank", "grade"],
}
POOL_KEYS = {
    "id":              ["id", "gacha_id", "oripa_id", "product_id"],
    "name":            ["name", "title"],
    "pack_price":      ["price", "pack_price", "unit_price"],
    "total_packs":     ["total", "total_packs", "pack_total", "quantity"],
    "remaining_packs": ["remaining", "remain", "remaining_packs", "stock"],
    "last_one_value":  ["last_one_value", "lastone", "last_prize_value"],
    "prizes":          ["prizes", "items", "cards", "lineup"],
}


def _pick(d: dict, keys: list, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _looks_like_pool(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    has_price = _pick(obj, POOL_KEYS["pack_price"]) is not None
    has_prizes = _pick(obj, POOL_KEYS["prizes"]) is not None
    return has_price and has_prizes


def _walk(obj, out):
    """JSONを再帰的に歩いてpoolっぽいdictを拾う"""
    if _looks_like_pool(obj):
        out.append(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def _parse_pool_dict(d: dict, site: str) -> OripaListing:
    raw_prizes = _pick(d, POOL_KEYS["prizes"], []) or []
    prizes = []
    for p in raw_prizes:
        if not isinstance(p, dict):
            continue
        total = int(_pick(p, PRIZE_KEYS["count_total"], 0) or 0)
        remain = _pick(p, PRIZE_KEYS["count_remaining"], None)
        if remain is None:
            remain = total
        prizes.append(PrizeItem(
            name=str(_pick(p, PRIZE_KEYS["name"], "unknown")),
            count_total=total,
            count_remaining=int(remain),
            stated_value=float(_pick(p, PRIZE_KEYS["stated_value"], 0) or 0),
            rarity=str(_pick(p, PRIZE_KEYS["rarity"], "") or ""),
        ))

    total_packs = int(_pick(d, POOL_KEYS["total_packs"], 0) or 0)
    remaining = _pick(d, POOL_KEYS["remaining_packs"], None)
    if remaining is None:
        remaining = total_packs

    return OripaListing(
        site=site,
        oripa_id=str(_pick(d, POOL_KEYS["id"], "")),
        name=str(_pick(d, POOL_KEYS["name"], "unknown")),
        pack_price=float(_pick(d, POOL_KEYS["pack_price"], 0) or 0),
        total_packs=total_packs,
        remaining_packs=int(remaining),
        prizes=prizes,
        is_transparent=True,
        is_finite=True,
        last_one_value=float(_pick(d, POOL_KEYS["last_one_value"], 0) or 0),
        url="",
        fetched_at=datetime.utcnow().isoformat(),
    )


class DopaScraper(OripaSiteScraper):
    SITE_NAME = "DOPA"
    BASE_URL = "https://dopa-game.jp"
    # 認証済みセッションで叩く候補エンドポイント (DevTools Network で特定する)
    CANDIDATE_ENDPOINTS = [
        "/api/gacha",
        "/api/gacha/list",
        "/api/v1/gacha",
        "/api/products",
        "/api/oripa/list",
    ]
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    REQUEST_DELAY = 2.0  # レート制限: サイトに優しく

    def __init__(self, mock_file: str = None, har_file: str = None,
                 cookie: str = None):
        """
        優先順位: mock_file > har_file > cookie live > 空
        cookie は省略時 env DOPA_COOKIE を参照
        """
        self.mock_file = mock_file
        self.har_file = har_file
        self.cookie = cookie or os.environ.get("DOPA_COOKIE", "")

    # ---------- 公開API ----------
    def fetch_pools(self) -> list:
        if self.mock_file and os.path.exists(self.mock_file):
            return self._fetch_from_mock()
        if self.har_file and os.path.exists(self.har_file):
            return self._fetch_from_har()
        if self.cookie:
            return self._fetch_live()
        print(f"[{self.SITE_NAME}] No source configured. "
              f"Provide --mock, --har, or set DOPA_COOKIE env.")
        return []

    # ---------- mock ----------
    def _fetch_from_mock(self) -> list:
        with open(self.mock_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        listings = []
        for item in data.get("oripas", []):
            prizes = [
                PrizeItem(
                    name=p["name"],
                    count_total=p["count_total"],
                    count_remaining=p.get("count_remaining", p["count_total"]),
                    stated_value=p["stated_value"],
                    rarity=p.get("rarity", ""),
                )
                for p in item.get("prizes", [])
            ]
            listings.append(OripaListing(
                site=self.SITE_NAME,
                oripa_id=item["id"],
                name=item["name"],
                pack_price=item["pack_price"],
                total_packs=item["total_packs"],
                remaining_packs=item.get("remaining_packs", item["total_packs"]),
                prizes=prizes,
                is_transparent=item.get("is_transparent", True),
                is_finite=item.get("is_finite", True),
                last_one_value=item.get("last_one_value", 0.0),
                url=item.get("url", ""),
                fetched_at=datetime.utcnow().isoformat(),
            ))
        return listings

    # ---------- HAR ----------
    def _fetch_from_har(self) -> list:
        """
        DevTools → Network → 右クリック "Save all as HAR with content"
        でエクスポートしたファイルを解析。dopa-game.jp のJSONレスポンスを
        全部舐めて pool っぽい dict を抽出する。
        """
        try:
            with open(self.har_file, "r", encoding="utf-8") as f:
                har = json.load(f)
        except Exception as e:
            print(f"[{self.SITE_NAME}] HAR parse error: {e}")
            return []

        entries = har.get("log", {}).get("entries", [])
        found = []
        for e in entries:
            req_url = e.get("request", {}).get("url", "")
            if "dopa-game.jp" not in req_url:
                continue
            resp = e.get("response", {}).get("content", {})
            mime = resp.get("mimeType", "")
            text = resp.get("text", "")
            if "json" not in mime or not text:
                continue
            try:
                body = json.loads(text)
            except Exception:
                continue
            _walk(body, found)

        # 重複排除 (id ベース)
        seen = set()
        listings = []
        for d in found:
            lid = str(_pick(d, POOL_KEYS["id"], id(d)))
            if lid in seen:
                continue
            seen.add(lid)
            try:
                listings.append(_parse_pool_dict(d, self.SITE_NAME))
            except Exception as ex:
                print(f"[{self.SITE_NAME}] skip malformed pool: {ex}")
        print(f"[{self.SITE_NAME}] HAR: {len(listings)} pools extracted "
              f"from {len(entries)} entries")
        return listings

    # ---------- live (authenticated) ----------
    def _http_get(self, path: str) -> tuple:
        url = self.BASE_URL + path
        req = urllib.request.Request(url, headers={
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ja,en;q=0.8",
            "Referer": self.BASE_URL + "/",
            "Cookie": self.cookie,
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, ""
        except Exception as e:
            return 0, str(e)

    def _fetch_live(self) -> list:
        """
        DOPA_COOKIE を使って候補エンドポイントを順に叩く。
        200でJSONが返ればpool抽出。全滅ならユーザに手順案内。
        """
        print(f"[{self.SITE_NAME}] live fetch with session cookie "
              f"(len={len(self.cookie)})")
        for path in self.CANDIDATE_ENDPOINTS:
            status, body = self._http_get(path)
            print(f"  {path} -> HTTP {status}")
            if status == 200 and body:
                try:
                    data = json.loads(body)
                except Exception:
                    continue
                found = []
                _walk(data, found)
                if found:
                    listings = []
                    for d in found:
                        try:
                            listings.append(_parse_pool_dict(d, self.SITE_NAME))
                        except Exception:
                            pass
                    print(f"[{self.SITE_NAME}] {len(listings)} pools via {path}")
                    return listings
            time.sleep(self.REQUEST_DELAY)

        print(f"[{self.SITE_NAME}] Live fetch failed. Recommended:")
        print("  1. Open DOPA in Chrome DevTools → Network → XHR")
        print("  2. Reload the gacha list page")
        print("  3. Right click → 'Save all as HAR with content'")
        print("  4. Run: python3 scanner.py --once --har <file>.har")
        return []
