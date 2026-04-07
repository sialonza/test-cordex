"""
EV計算エンジン

oripa_ev_analysis.py の数学モデルを scrapers/base.OripaListing に適用する。
"""
import sys
import os

# 親ディレクトリの oripa_ev_analysis.py をインポート
_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    from oripa_ev_analysis import PrizeTier, OripaPool
    HAS_EV_MODULE = True
except ImportError:
    HAS_EV_MODULE = False
    print("[WARN] oripa_ev_analysis.py not found, using fallback EV calc")


def to_oripa_pool(listing, use_market_value: bool = True):
    """OripaListing → OripaPool 変換"""
    if not HAS_EV_MODULE:
        return None

    tiers = []
    for p in listing.prizes:
        # 実勢価格があればそちらを優先、なければサイト提示値
        value = p.market_value if (use_market_value and p.market_value) else p.stated_value
        tiers.append(PrizeTier(
            name=p.name,
            market_value=float(value),
            count=p.count_remaining,  # 現在残っている枚数
        ))

    # 総売上 = 残口数 × 1口価格
    return OripaPool(
        name=listing.name,
        pack_price=listing.pack_price,
        total_packs=listing.remaining_packs,
        prize_tiers=tiers,
        last_one_prize_value=listing.last_one_value,
        is_infinite=not listing.is_finite,
    )


def compute_ev_simple(listing, use_market_value: bool = True) -> dict:
    """
    依存なしのフォールバック: 残枚数ベースの単純期待値計算

    EV = Σ (残枚数_i × 価値_i) / 残口数
    """
    total_value = 0.0
    for p in listing.prizes:
        v = p.market_value if (use_market_value and p.market_value) else p.stated_value
        total_value += p.count_remaining * v

    # ラストワン賞
    if listing.remaining_packs > 0:
        last_one_ev = listing.last_one_value / listing.remaining_packs
    else:
        last_one_ev = 0.0

    if listing.remaining_packs == 0:
        return {"ev": 0, "rtp": 0, "edge": 0, "is_positive": False,
                "total_value": 0, "remaining": 0}

    ev = (total_value / listing.remaining_packs) + last_one_ev
    rtp = ev / listing.pack_price if listing.pack_price > 0 else 0
    edge = rtp - 1.0  # プラスなら +EV

    return {
        "ev": ev,
        "rtp": rtp,
        "edge": edge,
        "is_positive": edge > 0,
        "total_value": total_value,
        "remaining": listing.remaining_packs,
    }


def compute_ev(listing, use_market_value: bool = True) -> dict:
    """統一エントリポイント"""
    # シンプル計算が信頼できる（透明プールのみ扱う前提）
    return compute_ev_simple(listing, use_market_value)


def format_ev_report(listing, ev: dict) -> str:
    """人間可読なEVレポート文字列を生成"""
    sign = "+" if ev["edge"] >= 0 else ""
    mark = "🟢" if ev["is_positive"] else "🔴"
    return (
        f"{mark} [{listing.site}] {listing.name}\n"
        f"   価格: ¥{listing.pack_price:,.0f} / 残{ev['remaining']}口\n"
        f"   期待値: ¥{ev['ev']:,.0f} (RTP {ev['rtp']*100:.1f}%, "
        f"{sign}{ev['edge']*100:.1f}%)"
    )
