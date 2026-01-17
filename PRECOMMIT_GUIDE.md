# Pre-commit セットアップガイド

このプロジェクトでは、CIと同じlintチェックをローカル環境のコミット前に自動実行できます。

## 📋 pre-commitとは

Git commitの前に自動的にコードチェックを実行するツールです。
CI/CDパイプラインと同じチェックをローカルで実行することで、コミット前に問題を発見できます。

## 🚀 セットアップ

### 1. インストール

```bash
# 開発用パッケージをインストール
pip install -r requirements-dev.txt

# pre-commitフックを有効化
pre-commit install
```

### 2. 初回実行（オプション）

既存のすべてのファイルに対してチェックを実行：

```bash
pre-commit run --all-files
```

## ✅ 実行されるチェック

コミット時に以下のチェックが自動実行されます：

### 1. **Black** - コードフォーマット
- Python コードを自動フォーマット
- 行の長さ: 最大127文字
- 自動修正: ✅

### 2. **Flake8** - Lint チェック
- コーディング規約違反を検出
- 複雑度チェック
- 自動修正: ❌（手動修正が必要）

### 3. **Mypy** - 型チェック
- 型ヒントの整合性をチェック
- 型エラーを検出
- 自動修正: ❌（手動修正が必要）

### 4. **Bandit** - セキュリティチェック
- セキュリティ上の問題を検出
- 脆弱なコードパターンを警告
- 自動修正: ❌（手動修正が必要）

### 5. **一般的なファイルチェック**
- 末尾の空白を削除
- ファイル末尾の改行を確保
- YAMLファイルの構文チェック
- 巨大ファイルの警告
- マージコンフリクトの検出

## 🔧 使い方

### 通常のコミット

```bash
git add .
git commit -m "your commit message"
```

コミット時に自動的にpre-commitフックが実行されます。

### チェックに失敗した場合

**自動修正された場合:**
```bash
# 修正されたファイルを再度追加
git add .
git commit -m "your commit message"
```

**手動修正が必要な場合:**
```bash
# エラーメッセージを確認して修正
vim file_with_error.py

# 修正後、再度コミット
git add .
git commit -m "your commit message"
```

### フックをスキップする（非推奨）

緊急時のみ使用してください：

```bash
git commit --no-verify -m "your commit message"
```

## 📝 手動実行

### 全ファイルをチェック

```bash
pre-commit run --all-files
```

### 特定のフックのみ実行

```bash
# Blackのみ
pre-commit run black --all-files

# Flake8のみ
pre-commit run flake8 --all-files

# Mypyのみ
pre-commit run mypy --all-files

# Banditのみ
pre-commit run bandit --all-files
```

### 特定のファイルのみチェック

```bash
pre-commit run --files main.py translation_service.py
```

## 🔄 フックの更新

`.pre-commit-config.yaml` を更新した後：

```bash
# フックを再インストール
pre-commit install

# 自動アップデート（最新バージョンに更新）
pre-commit autoupdate
```

## ⚙️ 設定ファイル

### `.pre-commit-config.yaml`
Pre-commitフックの設定ファイル。各チェックツールとバージョンを定義。

### `pyproject.toml`
Black、Mypy、Bandit、Pytestの設定を集約。

### `.flake8`
Flake8専用の設定ファイル。除外ディレクトリやルールの定義。

## 🆚 CI/CD との関係

| 項目 | Pre-commit (ローカル) | CI/CD (GitHub Actions) |
|------|----------------------|------------------------|
| 実行タイミング | コミット前 | プッシュ後 |
| チェック内容 | 同じ | 同じ |
| Python バージョン | ローカル環境 | 3.9-3.12 (マトリックス) |
| 目的 | 早期発見 | 最終検証 |

**Pre-commitを使うメリット:**
- ❌ CIで失敗してから気づく
- ✅ コミット前に問題を発見・修正
- ✅ CIの実行時間を短縮
- ✅ レビュアーの負担を軽減

## 🐛 トラブルシューティング

### フックが実行されない

```bash
# フックが正しくインストールされているか確認
ls -la .git/hooks/pre-commit

# 再インストール
pre-commit uninstall
pre-commit install
```

### チェックが遅い

```bash
# キャッシュをクリア
pre-commit clean
```

### 特定のファイルを除外したい

`.pre-commit-config.yaml` の各フックに `exclude:` を追加：

```yaml
- repo: https://github.com/psf/black
  rev: 23.12.1
  hooks:
    - id: black
      exclude: ^(venv|models|specific_file\.py)/
```

## 📚 参考リンク

- [Pre-commit 公式ドキュメント](https://pre-commit.com/)
- [Black ドキュメント](https://black.readthedocs.io/)
- [Flake8 ドキュメント](https://flake8.pycqa.org/)
- [Mypy ドキュメント](https://mypy.readthedocs.io/)
- [Bandit ドキュメント](https://bandit.readthedocs.io/)
