/**
 * BTC RSI Divergence Detection Engine
 * だまし排除に特化した多層フィルタリングシステム
 */

// ============================================================
// 1. RSI計算（Wilder's Smoothing - 正統派）
// ============================================================
function calcRSI(closes, period = 14) {
  if (closes.length < period + 1) return [];

  const rsi = new Array(closes.length).fill(null);
  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff > 0) avgGain += diff;
    else avgLoss += Math.abs(diff);
  }
  avgGain /= period;
  avgLoss /= period;

  rsi[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);

  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? Math.abs(diff) : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    rsi[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return rsi;
}

// ============================================================
// 2. ATR計算（ノイズ閾値に使用）
// ============================================================
function calcATR(highs, lows, closes, period = 14) {
  const atr = new Array(highs.length).fill(null);
  const tr = [];

  for (let i = 0; i < highs.length; i++) {
    if (i === 0) {
      tr.push(highs[i] - lows[i]);
    } else {
      tr.push(Math.max(
        highs[i] - lows[i],
        Math.abs(highs[i] - closes[i - 1]),
        Math.abs(lows[i] - closes[i - 1])
      ));
    }
  }

  let sum = 0;
  for (let i = 0; i < period; i++) sum += tr[i];
  atr[period - 1] = sum / period;

  for (let i = period; i < tr.length; i++) {
    atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period;
  }
  return atr;
}

// ============================================================
// 3. 出来高加重移動平均（VWMA）
// ============================================================
function calcVWMA(closes, volumes, period = 20) {
  const vwma = new Array(closes.length).fill(null);
  for (let i = period - 1; i < closes.length; i++) {
    let sumPV = 0, sumV = 0;
    for (let j = i - period + 1; j <= i; j++) {
      sumPV += closes[j] * volumes[j];
      sumV += volumes[j];
    }
    vwma[i] = sumV > 0 ? sumPV / sumV : closes[i];
  }
  return vwma;
}

// ============================================================
// 4. スイングポイント検出（ATRベースノイズフィルタ付き）
// ============================================================
function findSwingPoints(data, atr, lookback = 5, atrMultiplier = 0.3) {
  const swingHighs = [];
  const swingLows = [];

  for (let i = lookback; i < data.length - lookback; i++) {
    const currentATR = atr[i] || atr.find(v => v !== null) || 1;
    const threshold = currentATR * atrMultiplier;

    // スイングハイ検出
    let isHigh = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (data[j].high >= data[i].high - threshold) {
        if (j !== i && data[j].high > data[i].high) {
          isHigh = false;
          break;
        }
      }
    }
    // 左右の全バーより明確に高いか確認
    if (isHigh) {
      let leftMax = -Infinity, rightMax = -Infinity;
      for (let j = i - lookback; j < i; j++) leftMax = Math.max(leftMax, data[j].high);
      for (let j = i + 1; j <= i + lookback; j++) rightMax = Math.max(rightMax, data[j].high);
      if (data[i].high > leftMax && data[i].high > rightMax) {
        swingHighs.push({ index: i, price: data[i].high, time: data[i].timestamp });
      }
    }

    // スイングロー検出
    let isLow = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (data[j].low <= data[i].low + threshold) {
        if (j !== i && data[j].low < data[i].low) {
          isLow = false;
          break;
        }
      }
    }
    if (isLow) {
      let leftMin = Infinity, rightMin = Infinity;
      for (let j = i - lookback; j < i; j++) leftMin = Math.min(leftMin, data[j].low);
      for (let j = i + 1; j <= i + lookback; j++) rightMin = Math.min(rightMin, data[j].low);
      if (data[i].low < leftMin && data[i].low < rightMin) {
        swingLows.push({ index: i, price: data[i].low, time: data[i].timestamp });
      }
    }
  }

  return { swingHighs, swingLows };
}

// ============================================================
// 5. RSIスイングポイント検出
// ============================================================
function findRSISwings(rsiValues, lookback = 5) {
  const highs = [];
  const lows = [];

  for (let i = lookback; i < rsiValues.length - lookback; i++) {
    if (rsiValues[i] === null) continue;

    let isHigh = true, isLow = true;
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i || rsiValues[j] === null) continue;
      if (rsiValues[j] >= rsiValues[i]) isHigh = false;
      if (rsiValues[j] <= rsiValues[i]) isLow = false;
    }
    if (isHigh) highs.push({ index: i, value: rsiValues[i] });
    if (isLow) lows.push({ index: i, value: rsiValues[i] });
  }

  return { highs, lows };
}

// ============================================================
// 6. だまし排除フィルター群
// ============================================================

// フィルター1: RSI極限ゾーン確認（70/30を超えていること）
function filterRSIExtreme(rsiA, rsiB, type) {
  if (type === 'bearish') {
    return rsiA > 65 || rsiB > 65; // 少なくとも片方が過熱圏近く
  } else {
    return rsiA < 35 || rsiB < 35; // 少なくとも片方が売られすぎ圏近く
  }
}

// フィルター2: 出来高確認（2点目で出来高減少 = 勢い喪失）
function filterVolume(volumes, idxA, idxB, type) {
  const windowA = volumes.slice(Math.max(0, idxA - 2), idxA + 3);
  const windowB = volumes.slice(Math.max(0, idxB - 2), idxB + 3);
  const avgVolA = windowA.reduce((a, b) => a + b, 0) / windowA.length;
  const avgVolB = windowB.reduce((a, b) => a + b, 0) / windowB.length;

  // 弱気ダイバージェンス: 2点目で出来高が減少していれば上昇の勢い喪失
  if (type === 'bearish') return avgVolB < avgVolA * 1.2;
  // 強気ダイバージェンス: 2点目で出来高が減少していれば下落の勢い喪失
  return avgVolB < avgVolA * 1.2;
}

// フィルター3: 価格変動の有意性（ATR比でノイズを排除）
function filterSignificance(priceA, priceB, avgATR) {
  return Math.abs(priceA - priceB) > avgATR * 0.5;
}

// フィルター4: RSI変動の有意性（微小な差は無視）
function filterRSISignificance(rsiA, rsiB) {
  return Math.abs(rsiA - rsiB) > 3.0; // 3ポイント以上の差
}

// フィルター5: 2点間の距離（近すぎず遠すぎず）
function filterDistance(idxA, idxB, minBars = 5, maxBars = 40) {
  const dist = Math.abs(idxB - idxA);
  return dist >= minBars && dist <= maxBars;
}

// フィルター6: 中間でのRSI反転がないこと（きれいなダイバージェンス）
function filterNoIntermediateReversal(rsiValues, idxA, idxB, rsiA, rsiB, type) {
  for (let i = idxA + 1; i < idxB; i++) {
    if (rsiValues[i] === null) continue;
    if (type === 'bearish') {
      // 弱気: RSIが下がっているのに、中間でrsiB以上に上がっていたらNG
      if (rsiValues[i] > Math.max(rsiA, rsiB) + 2) return false;
    } else {
      // 強気: RSIが上がっているのに、中間でrsiB以下に下がっていたらNG
      if (rsiValues[i] < Math.min(rsiA, rsiB) - 2) return false;
    }
  }
  return true;
}

// ============================================================
// 7. ダイバージェンス検出（全フィルター統合）
// ============================================================
function detectDivergences(data, rsiValues, atr, volumes) {
  const { swingHighs, swingLows } = findSwingPoints(data, atr);
  const rsiSwings = findRSISwings(rsiValues);
  const results = [];

  // --- 弱気ダイバージェンス（価格↑ RSI↓ → 下落シグナル）---
  for (let i = 0; i < swingHighs.length - 1; i++) {
    for (let j = i + 1; j < swingHighs.length; j++) {
      const pA = swingHighs[i];
      const pB = swingHighs[j];

      if (pB.price <= pA.price) continue; // 価格は高値更新が必要

      // 対応するRSIスイングを探す（±2バー以内）
      const rsiA = findNearestRSISwing(rsiSwings.highs, pA.index, 3);
      const rsiB = findNearestRSISwing(rsiSwings.highs, pB.index, 3);
      if (!rsiA || !rsiB) continue;

      if (rsiB.value >= rsiA.value) continue; // RSIは下がっている必要

      // 全フィルター適用
      const avgATR = (atr[pA.index] || 0 + atr[pB.index] || 0) / 2;
      if (!filterDistance(pA.index, pB.index)) continue;
      if (!filterRSIExtreme(rsiA.value, rsiB.value, 'bearish')) continue;
      if (!filterSignificance(pA.price, pB.price, avgATR)) continue;
      if (!filterRSISignificance(rsiA.value, rsiB.value)) continue;
      if (!filterVolume(volumes, pA.index, pB.index, 'bearish')) continue;
      if (!filterNoIntermediateReversal(rsiValues, pA.index, pB.index, rsiA.value, rsiB.value, 'bearish')) continue;

      const confidence = calcConfidence({
        rsiA: rsiA.value, rsiB: rsiB.value,
        priceA: pA.price, priceB: pB.price,
        avgATR, type: 'bearish',
        distance: pB.index - pA.index,
        volumes, idxA: pA.index, idxB: pB.index
      });

      if (confidence < 60) continue; // 信頼度60%未満は排除

      results.push({
        type: 'bearish_regular',
        label: '弱気ダイバージェンス（下落警戒）',
        pointA: { index: pA.index, price: pA.price, rsi: rsiA.value, time: pA.time },
        pointB: { index: pB.index, price: pB.price, rsi: rsiB.value, time: pB.time },
        confidence
      });
    }
  }

  // --- 強気ダイバージェンス（価格↓ RSI↑ → 上昇シグナル）---
  for (let i = 0; i < swingLows.length - 1; i++) {
    for (let j = i + 1; j < swingLows.length; j++) {
      const pA = swingLows[i];
      const pB = swingLows[j];

      if (pB.price >= pA.price) continue; // 価格は安値更新が必要

      const rsiA = findNearestRSISwing(rsiSwings.lows, pA.index, 3);
      const rsiB = findNearestRSISwing(rsiSwings.lows, pB.index, 3);
      if (!rsiA || !rsiB) continue;

      if (rsiB.value <= rsiA.value) continue; // RSIは上がっている必要

      const avgATR = (atr[pA.index] || 0 + atr[pB.index] || 0) / 2;
      if (!filterDistance(pA.index, pB.index)) continue;
      if (!filterRSIExtreme(rsiA.value, rsiB.value, 'bullish')) continue;
      if (!filterSignificance(pA.price, pB.price, avgATR)) continue;
      if (!filterRSISignificance(rsiA.value, rsiB.value)) continue;
      if (!filterVolume(volumes, pA.index, pB.index, 'bullish')) continue;
      if (!filterNoIntermediateReversal(rsiValues, pA.index, pB.index, rsiA.value, rsiB.value, 'bullish')) continue;

      const confidence = calcConfidence({
        rsiA: rsiA.value, rsiB: rsiB.value,
        priceA: pA.price, priceB: pB.price,
        avgATR, type: 'bullish',
        distance: pB.index - pA.index,
        volumes, idxA: pA.index, idxB: pB.index
      });

      if (confidence < 60) continue;

      results.push({
        type: 'bullish_regular',
        label: '強気ダイバージェンス（上昇期待）',
        pointA: { index: pA.index, price: pA.price, rsi: rsiA.value, time: pA.time },
        pointB: { index: pB.index, price: pB.price, rsi: rsiB.value, time: pB.time },
        confidence
      });
    }
  }

  // --- ヒドゥン弱気（価格LH + RSI HH → トレンド継続下落）---
  for (let i = 0; i < swingHighs.length - 1; i++) {
    for (let j = i + 1; j < swingHighs.length; j++) {
      const pA = swingHighs[i];
      const pB = swingHighs[j];
      if (pB.price >= pA.price) continue; // 価格は切り下げ

      const rsiA = findNearestRSISwing(rsiSwings.highs, pA.index, 3);
      const rsiB = findNearestRSISwing(rsiSwings.highs, pB.index, 3);
      if (!rsiA || !rsiB) continue;
      if (rsiB.value <= rsiA.value) continue; // RSIは切り上げ

      const avgATR = (atr[pA.index] || 0 + atr[pB.index] || 0) / 2;
      if (!filterDistance(pA.index, pB.index)) continue;
      if (!filterSignificance(pA.price, pB.price, avgATR)) continue;
      if (!filterRSISignificance(rsiA.value, rsiB.value)) continue;

      const confidence = calcConfidence({
        rsiA: rsiA.value, rsiB: rsiB.value,
        priceA: pA.price, priceB: pB.price,
        avgATR, type: 'bearish',
        distance: pB.index - pA.index,
        volumes, idxA: pA.index, idxB: pB.index
      });

      if (confidence < 65) continue;

      results.push({
        type: 'bearish_hidden',
        label: 'ヒドゥン弱気（下降トレンド継続）',
        pointA: { index: pA.index, price: pA.price, rsi: rsiA.value, time: pA.time },
        pointB: { index: pB.index, price: pB.price, rsi: rsiB.value, time: pB.time },
        confidence
      });
    }
  }

  // --- ヒドゥン強気（価格HL + RSI LL → トレンド継続上昇）---
  for (let i = 0; i < swingLows.length - 1; i++) {
    for (let j = i + 1; j < swingLows.length; j++) {
      const pA = swingLows[i];
      const pB = swingLows[j];
      if (pB.price <= pA.price) continue; // 価格は切り上げ

      const rsiA = findNearestRSISwing(rsiSwings.lows, pA.index, 3);
      const rsiB = findNearestRSISwing(rsiSwings.lows, pB.index, 3);
      if (!rsiA || !rsiB) continue;
      if (rsiB.value >= rsiA.value) continue; // RSIは切り下げ

      const avgATR = (atr[pA.index] || 0 + atr[pB.index] || 0) / 2;
      if (!filterDistance(pA.index, pB.index)) continue;
      if (!filterSignificance(pA.price, pB.price, avgATR)) continue;
      if (!filterRSISignificance(rsiA.value, rsiB.value)) continue;

      const confidence = calcConfidence({
        rsiA: rsiA.value, rsiB: rsiB.value,
        priceA: pA.price, priceB: pB.price,
        avgATR, type: 'bullish',
        distance: pB.index - pA.index,
        volumes, idxA: pA.index, idxB: pB.index
      });

      if (confidence < 65) continue;

      results.push({
        type: 'bullish_hidden',
        label: 'ヒドゥン強気（上昇トレンド継続）',
        pointA: { index: pA.index, price: pA.price, rsi: rsiA.value, time: pA.time },
        pointB: { index: pB.index, price: pB.price, rsi: rsiB.value, time: pB.time },
        confidence
      });
    }
  }

  return results;
}

// ============================================================
// 8. 最近傍RSIスイング検索
// ============================================================
function findNearestRSISwing(rsiSwings, targetIndex, tolerance) {
  let best = null;
  let bestDist = Infinity;
  for (const s of rsiSwings) {
    const dist = Math.abs(s.index - targetIndex);
    if (dist <= tolerance && dist < bestDist) {
      best = s;
      bestDist = dist;
    }
  }
  return best;
}

// ============================================================
// 9. 信頼度スコア計算（0-100）
// ============================================================
function calcConfidence({ rsiA, rsiB, priceA, priceB, avgATR, type, distance, volumes, idxA, idxB }) {
  let score = 0;

  // (A) RSI乖離度 (max 25pt)
  const rsiDiff = Math.abs(rsiA - rsiB);
  score += Math.min(25, rsiDiff * 2.5);

  // (B) RSI極限ゾーン度 (max 20pt)
  if (type === 'bearish') {
    const maxRSI = Math.max(rsiA, rsiB);
    if (maxRSI > 75) score += 20;
    else if (maxRSI > 70) score += 15;
    else if (maxRSI > 65) score += 10;
    else score += 5;
  } else {
    const minRSI = Math.min(rsiA, rsiB);
    if (minRSI < 25) score += 20;
    else if (minRSI < 30) score += 15;
    else if (minRSI < 35) score += 10;
    else score += 5;
  }

  // (C) 価格変動のATR比 (max 20pt)
  if (avgATR > 0) {
    const priceDiffATR = Math.abs(priceA - priceB) / avgATR;
    score += Math.min(20, priceDiffATR * 5);
  }

  // (D) 2点間の距離の適正度 (max 15pt)
  if (distance >= 10 && distance <= 25) score += 15;
  else if (distance >= 7 && distance <= 35) score += 10;
  else score += 5;

  // (E) 出来高の裏付け (max 20pt)
  if (volumes && idxA !== undefined && idxB !== undefined) {
    const windowA = volumes.slice(Math.max(0, idxA - 1), idxA + 2);
    const windowB = volumes.slice(Math.max(0, idxB - 1), idxB + 2);
    const avgA = windowA.reduce((a, b) => a + b, 0) / windowA.length;
    const avgB = windowB.reduce((a, b) => a + b, 0) / windowB.length;
    const volRatio = avgA > 0 ? avgB / avgA : 1;

    // 勢い喪失（出来高減少）の確認
    if (volRatio < 0.7) score += 20;
    else if (volRatio < 0.9) score += 15;
    else if (volRatio < 1.1) score += 10;
    else score += 5;
  }

  return Math.min(100, Math.round(score));
}

// ============================================================
// 10. マルチタイムフレーム合流分析
// ============================================================
function multiTimeframeAnalysis(results1h, results4h, results1D) {
  const signals = [];

  // 各タイムフレームの最新シグナルを取得
  const latest = {
    '1h': getLatestByType(results1h),
    '4h': getLatestByType(results4h),
    '1D': getLatestByType(results1D)
  };

  // 合流判定
  const directions = ['bullish', 'bearish'];
  for (const dir of directions) {
    const matching = [];
    for (const [tf, signals_] of Object.entries(latest)) {
      if (signals_[dir]) matching.push({ tf, signal: signals_[dir] });
    }

    if (matching.length >= 2) {
      // 2つ以上のタイムフレームで一致 → 高信頼シグナル
      const avgConf = matching.reduce((s, m) => s + m.signal.confidence, 0) / matching.length;
      const bonus = matching.length === 3 ? 15 : 5;

      signals.push({
        direction: dir,
        confluence: matching.length,
        timeframes: matching.map(m => m.tf),
        avgConfidence: Math.min(100, Math.round(avgConf + bonus)),
        details: matching.map(m => ({
          timeframe: m.tf,
          type: m.signal.type,
          confidence: m.signal.confidence,
          pointB_time: m.signal.pointB.time
        })),
        grade: matching.length === 3 ? 'S' : 'A'
      });
    } else if (matching.length === 1) {
      const m = matching[0];
      // 上位足のみは参考情報として出力
      if (m.tf === '1D' && m.signal.confidence >= 75) {
        signals.push({
          direction: dir,
          confluence: 1,
          timeframes: [m.tf],
          avgConfidence: m.signal.confidence,
          details: [{ timeframe: m.tf, type: m.signal.type, confidence: m.signal.confidence, pointB_time: m.signal.pointB.time }],
          grade: 'B'
        });
      }
    }
  }

  return signals;
}

function getLatestByType(results) {
  const latest = {};
  for (const r of results) {
    const dir = r.type.includes('bullish') ? 'bullish' : 'bearish';
    if (!latest[dir] || r.pointB.index > latest[dir].pointB.index) {
      latest[dir] = r;
    }
  }
  return latest;
}

// ============================================================
// 11. メインの分析パイプライン
// ============================================================
function analyzeCandles(candles) {
  // データ整形（時系列昇順に）
  const sorted = [...candles].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  const closes = sorted.map(c => parseFloat(c.close));
  const highs = sorted.map(c => parseFloat(c.high));
  const lows = sorted.map(c => parseFloat(c.low));
  const volumes = sorted.map(c => parseFloat(c.volume));

  // 指標計算
  const rsi = calcRSI(closes);
  const atr = calcATR(highs, lows, closes);

  // ダイバージェンス検出
  const divergences = detectDivergences(sorted, rsi, atr, volumes);

  return { candles: sorted, closes, highs, lows, volumes, rsi, atr, divergences };
}

// ============================================================
// 12. エクスポート
// ============================================================
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    calcRSI, calcATR, calcVWMA, findSwingPoints, findRSISwings,
    detectDivergences, multiTimeframeAnalysis, analyzeCandles, calcConfidence
  };
}
