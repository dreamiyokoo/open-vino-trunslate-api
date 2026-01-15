# OpenVINO Translation API

OpenVINOを使った高速AI翻訳WebサービスとAPI

## 🚀 特徴

- **OpenVINO最適化**: Intel OpenVINOによる高速推論
- **多言語対応**: 英語、日本語、中国語、韓国語、フランス語、ドイツ語、スペイン語、ロシア語
- **自動言語検出**: 入力言語を自動で検出
- **REST API**: プログラマティックなアクセス用のAPI
- **モダンUI**: レスポンシブでモダンなWebインターフェース
- **VS Code統合**: launch.jsonでデバッグ実行可能

## 📋 必要な環境

- Python 3.8以上
- 十分なメモリ (モデルサイズに依存、最低4GB推奨)

## 🛠️ セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. アプリケーションの起動

#### コマンドラインから起動

```bash
python main.py
```

または

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### VS Codeから起動

1. F5キーを押す
2. "FastAPI: OpenVINO Translation" を選択

### 3. アクセス

- Webインターフェース: http://localhost:8000
- APIドキュメント: http://localhost:8000/docs
- ReDocドキュメント: http://localhost:8000/redoc

## 📡 API使用方法

### 翻訳エンドポイント

```bash
POST /api/translate
Content-Type: application/json

{
  "text": "Hello, world!",
  "target_lang": "ja",
  "source_lang": "en"  // オプション
}
```

#### レスポンス例

```json
{
  "translated_text": "こんにちは、世界！",
  "source_lang": "en",
  "target_lang": "ja",
  "original_text": "Hello, world!"
}
```

### サポート言語の取得

```bash
GET /api/languages
```

### ヘルスチェック

```bash
GET /api/health
```

## 🌐 サポートされている言語

| コード | 言語 |
|--------|------|
| en | English (英語) |
| ja | Japanese (日本語) |
| zh | Chinese (中国語) |
| ko | Korean (韓国語) |
| fr | French (フランス語) |
| de | German (ドイツ語) |
| es | Spanish (スペイン語) |
| ru | Russian (ロシア語) |

## 📁 プロジェクト構造

```
open-vino-trunslate-api/
├── main.py                    # FastAPIアプリケーション
├── translation_service.py     # OpenVINO翻訳サービス
├── requirements.txt           # 依存パッケージ
├── requirements-dev.txt       # 開発用パッケージ
├── pytest.ini                 # pytestの設定
├── .coveragerc                # カバレッジ設定
├── README.md                  # このファイル
├── .vscode/
│   └── launch.json           # VS Codeデバッグ設定
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD
├── templates/
│   └── index.html            # Webインターフェース
├── static/
│   └── style.css             # スタイルシート
├── tests/
│   ├── test_main.py          # APIエンドポイントのテスト
│   └── test_translation_service.py  # 翻訳サービスのテスト
└── models/                   # モデルキャッシュ (自動生成)
```

## 🎨 Webインターフェース

モダンで直感的なUIを提供:

- **自動言語検出**: 入力言語を自動で判定
- **リアルタイム翻訳**: ボタンクリックで即座に翻訳
- **コピー機能**: 翻訳結果をワンクリックでコピー
- **レスポンシブデザイン**: PC・タブレット・スマホに対応
- **アニメーション**: スムーズなUIアニメーション

## ⚙️ 設定

### モデルキャッシュディレクトリの変更

`translation_service.py`の`TranslationService`クラスで設定可能:

```python
service = TranslationService(cache_dir="./custom_models")
```

### ポート番号の変更

`main.py`の最後の部分を編集:

```python
uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
```

## 🔧 トラブルシューティング

### モデルのダウンロードが遅い場合

初回起動時、Hugging Faceからモデルをダウンロードします。
環境変数を設定してミラーサイトを使用できます:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### メモリ不足エラー

複数のモデルを同時に使用するとメモリを消費します。
使用しない言語ペアのモデルは削除するか、より多くのメモリを確保してください。

## 🧪 テストの実行

### テスト環境のセットアップ

```bash
pip install -r requirements-dev.txt
```

### テストの実行

```bash
# 全テストを実行
pytest

# カバレッジ付きで実行
pytest --cov=. --cov-report=html

# 特定のテストファイルのみ実行
pytest tests/test_main.py -v

# 詳細な出力で実行
pytest -v -s
```

### コードフォーマットとリント

```bash
# コードフォーマットのチェック
black --check .

# コードフォーマットの自動修正
black .

# flake8でリント
flake8 .

# 型チェック
mypy main.py translation_service.py
```

## 🚀 CI/CD

GitHub Actionsによる自動テストとデプロイメントパイプラインを実装しています。

- **自動テスト**: プッシュとプルリクエスト時に自動実行
- **マルチバージョンテスト**: Python 3.9-3.12でテスト
- **コードカバレッジ**: Codecovにレポート送信
- **セキュリティスキャン**: Trivyによる脆弱性チェック
- **コード品質**: Black、Flake8による自動チェック

ワークフローファイル: [.github/workflows/ci.yml](.github/workflows/ci.yml)

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 使用技術

- [FastAPI](https://fastapi.tiangolo.com/) - モダンなPython Webフレームワーク
- [OpenVINO](https://docs.openvino.ai/) - Intel AI推論最適化ツールキット
- [Hugging Face Transformers](https://huggingface.co/transformers/) - 事前学習済みモデル
- [Helsinki-NLP/OPUS-MT](https://huggingface.co/Helsinki-NLP) - 翻訳モデル
