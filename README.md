# テキスト整形アプリ (GitHub Pages 版)

入力テキストを「AI 風」に整形して返す、純粋な HTML/JavaScript のシングルページアプリです。ビルドや依存インストールは不要で、そのまま GitHub Pages へ公開できます。

## 収録ファイル
- `index.html` : アプリ本体 (スタイルとスクリプトを1ファイルに同梱)
- `.github/workflows/deploy-pages.yml` : GitHub Pages へ自動デプロイするワークフロー
- `.github/workflows/package-zip.yml` : ZIP アーカイブを生成して Artifact にアップロードするワークフロー

## ローカルで試す
1. このリポジトリを clone または ZIP 展開します。
2. `index.html` をブラウザで開くだけで動作します (HTTP サーバーは不要)。
3. テキストを入力し、**送信** ボタンまたは `⌘/Ctrl + Enter` で送信すると、数百ミリ秒で整形結果が表示されます。

> 返却テキストはローカル JavaScript で生成しています。外部の AI API には接続しません。

## GitHub Pages で公開する手順
1. GitHub 上でリポジトリを **Public** に設定します。
2. `main` ブランチへ push します (または Actions で `Deploy static content to Pages` を手動実行)。
3. Actions タブでワークフロー完了を待つと、Pages が有効になります。
4. 公開 URL は `https://<GitHubユーザー名>.github.io/<リポジトリ名>/` です。
   - 例: このリポジトリ名なら `https://<GitHubユーザー名>.github.io/test-cordex/`
5. デプロイ後、`index.html` 内の「公開URL」欄にも実際の公開先が表示されます。コピーして共有できます。

## ZIP ファイルの取得
### GitHub Actions Artifact からダウンロード
1. Actions タブで **Package app as zip** を開きます。
2. `main` ブランチの実行 (または手動実行) の Artifacts にある `formatted-text-app-zip` をダウンロードします。

### 手元で作成する
リポジトリルートで以下を実行すると、同等の ZIP を作成できます。

```bash
zip -r artifacts/app.zip index.html README.md .github/workflows/deploy-pages.yml .github/workflows/package-zip.yml
```

ZIP を配布すれば、ブラウザで `index.html` を開くだけで同じデモを利用できます。
