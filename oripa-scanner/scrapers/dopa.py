"""
DOPA スクレイパー

DOPAは最大手のオンラインオリパサイト。
商品ページは JavaScript レンダリングが多いので、APIエンドポイントを探す方が確実。
実際の実装では:
  1. ブラウザ DevTools で API エンドポイントを特定
  2. requests + Cookie / ヘッダー偽装で取得
  3. BeautifulSoup でフォールバック

このMVPではモックデータを返す（実サイトの実装は利用規約との兼ね合い）。
ユーザーは Chrome拡張や手動で .json をエクスポートして読み込ませる方式も可能。
"""
import json
import os
from datetime import datetime
from .base import OripaSiteScraper, OripaListing, PrizeItem


class DopaScraper(OripaSiteScraper):
    SITE_NAME = "DOPA"

    def __init__(self, mock_file: str = None):
        """
        mock_file: JSONファイルからプール情報を読み込む（開発・テスト用）
        本番では self._fetch_live() を使う
        """
        self.mock_file = mock_file

    def fetch_pools(self) -> list:
        if self.mock_file and os.path.exists(self.mock_file):
            return self._fetch_from_mock()
        return self._fetch_live()

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

    def _fetch_live(self) -> list:
        """
        実サイトからのスクレイピング実装

        TODO: DOPAのAPIエンドポイントを特定
        候補:
          - GET https://dopa-game.jp/api/oripa/list
          - GraphQL endpoint の可能性も
        必要な偽装:
          - User-Agent
          - Cookie (セッション)
          - Referer
        """
        import urllib.request
        import urllib.error

        # プレースホルダ: 実装時はここにAPIコール
        print(f"[{self.SITE_NAME}] Live scraping not implemented yet")
        print(f"[{self.SITE_NAME}] Use mock_file= or implement _fetch_live()")
        return []
