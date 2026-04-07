"""
信頼性レイヤ: ネットオリパは運営が確率操作しうるため、
計算EVをそのまま信じない。実測RTPとの乖離を追跡し、
サイト別信頼係数で割り引いた「実効EV」を算出する。

保存先: data/purchase_log.json
"""
import json
import os
import math
from datetime import datetime
from typing import Optional

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "purchase_log.json")


def _load() -> dict:
    if not os.path.exists(LOG_PATH):
        return {"purchases": [], "trust": {}}
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"purchases": [], "trust": {}}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_purchase(site: str, oripa_id: str, oripa_name: str,
                    pack_price: float, packs_bought: int,
                    calc_ev_per_pack: float,
                    actual_return: float,
                    notes: str = "") -> None:
    """
    1回の購入結果を記録。
      actual_return: 獲得物の実勢価格合計 (¥)
    """
    data = _load()
    data["purchases"].append({
        "ts": datetime.utcnow().isoformat(),
        "site": site,
        "oripa_id": oripa_id,
        "oripa_name": oripa_name,
        "pack_price": pack_price,
        "packs_bought": packs_bought,
        "spent": pack_price * packs_bought,
        "calc_ev_per_pack": calc_ev_per_pack,
        "calc_return": calc_ev_per_pack * packs_bought,
        "actual_return": actual_return,
        "notes": notes,
    })
    _save(data)


def site_stats(site: str) -> dict:
    """
    サイト別の 計算EV合計 vs 実測リターン合計 を集計し、
    χ²近似で「操作疑い度」を出す。
    """
    data = _load()
    rows = [p for p in data["purchases"] if p["site"] == site]
    if not rows:
        return {"site": site, "n": 0, "trust": 1.0, "note": "no data"}

    spent = sum(p["spent"] for p in rows)
    calc_return = sum(p["calc_return"] for p in rows)
    actual_return = sum(p["actual_return"] for p in rows)

    calc_rtp = calc_return / spent if spent > 0 else 0
    actual_rtp = actual_return / spent if spent > 0 else 0

    # 乖離: 実測 / 計算 (1.0 が完全一致)
    if calc_return > 0:
        ratio = actual_return / calc_return
    else:
        ratio = 1.0

    # χ²近似 (簡易): (observed - expected)^2 / expected
    chi2 = ((actual_return - calc_return) ** 2 / calc_return) if calc_return > 0 else 0

    # 信頼係数: サンプル数と乖離から決定
    #   n<5        : 仮置き 0.7
    #   ratio>=0.9 : 1.0 (健全)
    #   ratio<0.9  : 下方修正 (実測が計算より低い = 裏で何か)
    #   ratio>1.5  : 0.6 (逆に高すぎも罠サクラ疑い)
    n = len(rows)
    if n < 5:
        trust = 0.7
        note = f"insufficient samples (n={n})"
    elif ratio >= 1.5:
        trust = 0.6
        note = "suspiciously high actual RTP (sample bias?)"
    elif ratio >= 0.9:
        trust = min(1.0, 0.8 + 0.2 * (n / 20))
        note = "healthy"
    elif ratio >= 0.7:
        trust = 0.75 * ratio
        note = "mild underperformance"
    else:
        trust = 0.5 * ratio
        note = "significant deviation — possible manipulation"

    return {
        "site": site,
        "n": n,
        "spent": spent,
        "calc_rtp": calc_rtp,
        "actual_rtp": actual_rtp,
        "ratio": ratio,
        "chi2": chi2,
        "trust": round(trust, 3),
        "note": note,
    }


def effective_ev(site: str, calc_ev: float, calc_rtp: float) -> dict:
    """
    信頼係数で割り引いた実効EVを返す。
    さらに計算RTPが異常に高い場合は赤旗を立てる。
    """
    stats = site_stats(site)
    trust = stats["trust"]

    eff_ev = calc_ev * trust
    eff_rtp = calc_rtp * trust

    flags = []
    if calc_rtp > 2.0:
        flags.append("⚠️ calc RTP > 200% — likely bait/trap")
    if calc_rtp > 1.5:
        flags.append("⚠️ calc RTP > 150% — verify residual counts")
    if stats["n"] >= 5 and stats["ratio"] < 0.7:
        flags.append("⚠️ site under-delivers historically")

    return {
        "calc_ev": calc_ev,
        "calc_rtp": calc_rtp,
        "trust": trust,
        "effective_ev": eff_ev,
        "effective_rtp": eff_rtp,
        "flags": flags,
        "site_note": stats["note"],
    }
