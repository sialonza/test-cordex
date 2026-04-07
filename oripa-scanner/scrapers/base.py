"""
オリパサイト共通インターフェース

各スクレイパーは fetch_pools() を実装し、OripaListing のリストを返す。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PrizeItem:
    """プール内の1つの賞品（カード）"""
    name: str                    # "リザードン SAR"
    count_total: int             # プール内の総数
    count_remaining: int         # 残数（非透明プールの場合は count_total と同じ）
    stated_value: float          # サイトが提示する価値（円）
    market_value: Optional[float] = None  # 実勢価格（後から埋める）
    rarity: str = ""             # "SAR", "UR" 等


@dataclass
class OripaListing:
    """1つのオリパ商品"""
    site: str                    # "DOPA"
    oripa_id: str                # サイト内ID
    name: str                    # "ポケモン激アツオリパ Vol.3"
    pack_price: float            # 1口の価格（円）
    total_packs: int             # 総口数
    remaining_packs: int         # 残口数
    prizes: list = field(default_factory=list)  # list[PrizeItem]
    is_transparent: bool = False  # 残数が公開されているか
    is_finite: bool = True        # 有限プールか（False=無限供給）
    last_one_value: float = 0.0   # ラストワン賞の価値
    url: str = ""
    fetched_at: str = ""          # ISO timestamp


class OripaSiteScraper:
    """共通スクレイパーインターフェース"""
    SITE_NAME = "BASE"

    def fetch_pools(self) -> list:
        """現在アクティブなオリパ一覧を取得"""
        raise NotImplementedError

    def fetch_detail(self, oripa_id: str) -> OripaListing:
        """特定オリパの詳細情報を取得"""
        raise NotImplementedError
