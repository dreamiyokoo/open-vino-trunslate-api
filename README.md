# OpenVINO Translation & Chat API

OpenVINOを使った高速AI翻訳WebサービスとチャットAPI

## 🚀 特徴

- **OpenVINO最適化**: Intel OpenVINOによる高速推論
- **多言語翻訳対応**: 英語、日本語、中国語、韓国語、フランス語、ドイツ語、スペイン語、ロシア語、タイ語
- **AIチャット機能**: OpenVINO対応の軽量LLMを使用したチャット機能
- **統合UI**: 翻訳とチャットを切り替えられるタブインターフェース
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

### チャットエンドポイント

#### チャットメッセージ送信

```bash
POST /api/chat
Content-Type: application/json

{
  "message": "こんにちは",
  "session_id": "optional-session-id",
  "system_prompt": "あなたは親切なアシスタントです",
  "translate_to": "en"  // オプション: 応答を翻訳
}
```

#### レスポンス例

```json
{
  "response": "こんにちは！どのようにお手伝いできますか？",
  "session_id": "generated-or-provided-id",
  "timestamp": "2026-01-17T12:00:00Z",
  "translated_response": "Hello! How can I help you?"
}
```

#### チャット履歴の取得

```bash
GET /api/chat/history/{session_id}
```

#### チャット履歴の削除

```bash
DELETE /api/chat/history/{session_id}
```

#### アクティブセッション一覧

```bash
GET /api/chat/sessions
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
| th | Thai (タイ語) |

## 📁 プロジェクト構造

```
open-vino-trunslate-api/
├── main.py                    # FastAPIアプリケーション（翻訳＋チャット）
├── translation_service.py     # OpenVINO翻訳サービス
├── chat_service.py            # OpenVINOチャットサービス（新規）
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
│   └── index.html            # 統合Webインターフェース（翻訳＋チャット）
├── static/
│   └── style.css             # スタイルシート
├── tests/
│   ├── test_main.py          # APIエンドポイントのテスト
│   ├── test_translation_service.py  # 翻訳サービスのテスト
│   └── test_chat_service.py  # チャットサービスのテスト（新規）
└── models/                   # モデルキャッシュ (自動生成)
    ├── Helsinki-NLP_opus-mt-*  # 翻訳モデル
    └── chat_llm/             # チャット用LLMモデル（新規）
```

## 🎨 Webインターフェース

モダンで直感的なUIを提供:

- **タブ切り替え**: 翻訳とチャット機能を簡単に切り替え
- **翻訳機能**:
  - 自動言語検出
  - リアルタイム翻訳
  - コピー機能
- **チャット機能**:
  - AIとの対話形式のチャット
  - 会話履歴の保持
  - セッション管理
  - チャットバブル形式の表示
- **レスポンシブデザイン**: PC・タブレット・スマホに対応
- **アニメーション**: スムーズなUIアニメーション

## ⚙️ 設定

### モデルキャッシュディレクトリの変更

`translation_service.py`の`TranslationService`クラスで設定可能:

```python
service = TranslationService(cache_dir="./custom_models")
```

### チャット用LLMモデルの変更

`chat_service.py`の`ChatService`クラスでモデルを指定可能:

```python
service = ChatService(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    cache_dir="./models/chat_llm"
)
```

#### サポートされているLLMモデル

| モデル | サイズ | 特徴 | 推奨度 |
|--------|--------|------|--------|
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | 軽量、英語メイン（デフォルト） | ⭐⭐⭐ |
| Phi-2 | 2.7B | Microsoft製、高品質、英語 | ⭐⭐⭐⭐ |
| rinna/japanese-gpt-neox-small | 3.6B | 日本語特化 | ⭐⭐⭐⭐ |
| Qwen1.5-1.8B | 1.8B | 多言語対応 | ⭐⭐⭐ |

**注意**: モデルを変更する場合は、そのモデルのプロンプトフォーマットに合わせて`_format_prompt`メソッドを調整する必要があります。

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

### Pre-commitフックの使用（推奨）

Pre-commitフックを使用すると、コミット前に自動的にCIと同じlintチェックを実行できます。

```bash
# pre-commitのインストール（初回のみ）
pip install pre-commit

# pre-commitフックを有効化
pre-commit install

# 手動で全ファイルをチェック
pre-commit run --all-files

# 特定のフックのみ実行
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

**有効化後は、`git commit`時に自動的に以下がチェックされます:**
- ✅ Black（コードフォーマット）
- ✅ Flake8（Lintチェック）
- ✅ Mypy（型チェック）
- ✅ Bandit（セキュリティチェック）
- ✅ 一般的なファイルチェック（trailing whitespace等）

**チェックに失敗した場合:**
- 自動修正可能な問題は自動的に修正されます
- 再度コミットを試みてください
- 手動修正が必要な問題はメッセージに表示されます


## 🚀 CI/CD

GitHub Actionsによる自動テストとデプロイメントパイプラインを実装しています。

- **自動テスト**: プッシュとプルリクエスト時に自動実行
- **マルチバージョンテスト**: Python 3.9-3.12でテスト
- **コードカバレッジ**: Codecovにレポート送信
- **セキュリティスキャン**: Trivyによる脆弱性チェック
- **コード品質**: Black、Flake8による自動チェック

ワークフローファイル: [.github/workflows/ci.yml](.github/workflows/ci.yml)

## 📄 ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。詳細はLICENSEファイルをご覧ください。

## 🙏 使用技術

- [FastAPI](https://fastapi.tiangolo.com/) - モダンなPython Webフレームワーク
- [OpenVINO](https://docs.openvino.ai/) - Intel AI推論最適化ツールキット
- [Hugging Face Transformers](https://huggingface.co/transformers/) - 事前学習済みモデル
- [Helsinki-NLP/OPUS-MT](https://huggingface.co/Helsinki-NLP) - 翻訳モデル
- [TinyLlama](https://huggingface.co/TinyLlama) - 軽量チャット用LLM

## 🔧 LangChain対応（オプション）

このAPIはLangChainとの統合もサポートしています（オプション機能）。

### インストール

```bash
pip install langchain
```

### 使用方法

```python
from langchain_adapter import create_langchain_chat

# LangChain互換のチャットインスタンスを作成
chat = create_langchain_chat(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    system_prompt="あなたは親切なアシスタントです"
)

# チャットを実行
response = chat("こんにちは！")
print(response)

# 履歴を取得
history = chat.get_history()
print(history)

# 履歴をクリア
chat.clear_history()
```

### LangChainとの統合

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_adapter import OpenVINOChatLangChain

# OpenVINOチャットをLangChainで使用
llm = OpenVINOChatLangChain()

# PromptTemplateと組み合わせる
template = "あなたは{role}です。質問: {question}"
prompt = PromptTemplate(template=template, input_variables=["role", "question"])
chain = LLMChain(llm=llm, prompt=prompt)

# チェーンを実行
result = chain.run(role="親切なアシスタント", question="こんにちは")
print(result)
```

### API経由でのLangChain対応

APIエンドポイントでは`use_langchain`パラメータを将来の拡張用に予約していますが、
現在は内部実装はLangChainに依存せず、シンプルなPython実装で動作します。
これにより、LangChainを使わないユーザーも軽量に利用できます。
