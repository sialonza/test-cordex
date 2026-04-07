# オリパ期待値スキャナー

ネットオリパの期待値（EV）を自動計算し、+EVのオリパを検出する自分用ツール。

## アーキテクチャ

```
[scrapers/]        オリパサイトからプール情報を取得
  ├ dopa.py        DOPA（最大手）
  ├ clove.py       Clove
  └ base.py        共通インターフェース

[pricing.py]       メルカリ/ヤフオク実勢価格取得
[ev_engine.py]     期待値計算（oripa_ev_analysis.pyを利用）
[scanner.py]       メインスキャナー (CLI)
[notify.py]        +EV検出時のSlack/LINE通知

[data/]            収集データのキャッシュ
[logs/]            実行ログ
```

## 使い方

```bash
# 1回だけスキャン
python3 scanner.py --once

# 継続監視（5分間隔）
python3 scanner.py --watch --interval 300

# 特定オリパのEV詳細表示
python3 scanner.py --inspect <oripa_id>
```

## 注意

- 各サイトの利用規約を確認すること
- スクレイピング間隔を開けること（レート制限）
- これは分析ツールであり、購入推奨ではない
- ギャンブルは自己責任

## 数学的根拠

`../oripa_ev_analysis.py` の期待値フレームワークを利用。
有限プール型オリパでは残り枚数から条件付きEVを計算できる。
