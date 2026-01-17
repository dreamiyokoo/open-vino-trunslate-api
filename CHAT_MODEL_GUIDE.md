# Chat用LLMモデル選択ガイド

## 🤔 問題: TinyLlamaでの日本語応答が不自然

TinyLlama-1.1B-Chat-v1.0は英語メインのモデルのため、日本語での会話が不自然になります。

### 症状
- 「こんにちは」→ 意味不明な応答
- 文法がおかしい
- コンテキストを理解していない

## ✅ 解決策: 日本語対応LLMモデルを使用

#### 推奨モデル

| モデル | サイズ | 特徴 | 推奨度 |
|--------|--------|------|--------|
| **rinna/japanese-gpt-neox-small** | 3.6B | 日本語特化、会話対応 | ⭐⭐⭐⭐⭐ |
| **cyberagent/open-calm-small** | 1.4B | 日本語、軽量 | ⭐⭐⭐⭐ |
| **rinna/japanese-gpt-1b** | 1.4B | 日本語、バランス良い | ⭐⭐⭐⭐ |
| **stabilityai/japanese-stablelm-base-alpha-7b** | 7B | 高品質（要メモリ） | ⭐⭐⭐ |

#### 実装方法

**1. main.pyを編集**

```python
# main.py の get_chat_service() 関数を修正

def get_chat_service():
    global chat_service

    if chat_service is None:
        with chat_service_lock:
            if chat_service is None:
                # 日本語対応モデルを使用
                chat_service = ChatService(
                    model_name="rinna/japanese-gpt-neox-small"
                )
                logger.info("Chat service initialized with Japanese LLM")

    return chat_service
```

**2. chat_service.pyのプロンプトフォーマットを調整**

日本語モデルによってプロンプトフォーマットが異なる場合があります。

```python
def _format_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """モデルに応じてフォーマットを変更"""

    # rinnaモデルの場合
    if "rinna" in self.model_name.lower():
        # rinnaモデル用のフォーマット
        formatted = []
        if system_prompt:
            formatted.append(f"システム: {system_prompt}\n")

        for msg in messages:
            if msg["role"] == "user":
                formatted.append(f"ユーザー: {msg['content']}\n")
            elif msg["role"] == "assistant":
                formatted.append(f"アシスタント: {msg['content']}\n")

        formatted.append("アシスタント: ")
        return "".join(formatted)

    # TinyLlama等のデフォルト
    else:
        # 既存のフォーマット
        ...
```

**3. 初回実行時**

モデルのダウンロードと変換に時間がかかります（5-15分程度）。

```bash
# サーバー起動
python main.py

# 初回はモデルダウンロード
# INFO: Downloading model: rinna/japanese-gpt-neox-small
# INFO: Converting to OpenVINO format...
# INFO: Model loaded successfully
```

### オプション2: 翻訳を組み合わせる

日本語入力を英語に翻訳 → TinyLlamaで処理 → 日本語に翻訳

**実装例**:

```python
# chat_service.py に追加

def chat_with_translation(self, message: str, session_id: Optional[str] = None):
    """日本語メッセージを翻訳してから処理"""
    from translation_service import TranslationService

    translator = TranslationService()

    # 1. 日本語→英語
    english_message = translator.translate(message, target_lang="en", source_lang="ja")["translated_text"]

    # 2. 英語でチャット
    response = self.chat(english_message, session_id)

    # 3. 英語→日本語
    japanese_response = translator.translate(response["response"], target_lang="ja", source_lang="en")["translated_text"]

    response["response"] = japanese_response
    return response
```

## 🚀 クイックスタート

```bash
# 1. main.py を編集（上記参照）
vim main.py

# 2. サーバー起動（初回はダウンロードに時間がかかる）
python main.py

# 3. テスト
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "こんにちは"}'
```

## 📊 比較表

| 方式 | 品質 | 速度 | メモリ | セットアップ |
|------|------|------|--------|-------------|
| TinyLlama (英語) | ⭐ | ⭐⭐⭐⭐ | 2GB | 中 |
| 日本語LLM (1-2B) | ⭐⭐⭐⭐ | ⭐⭐⭐ | 4-6GB | 中 |
| 日本語LLM (7B) | ⭐⭐⭐⭐⭐ | ⭐⭐ | 10-14GB | 長 |
| 翻訳併用 | ⭐⭐⭐ | ⭐⭐ | 4GB | 中 |

## 🔧 トラブルシューティング

### メモリ不足

```python
# より小さいモデルを使用
chat_service = ChatService(
    model_name="cyberagent/open-calm-small"  # 1.4B
)
```

### ダウンロードが遅い

```bash
# Hugging Face CLIで事前ダウンロード
pip install huggingface-hub
huggingface-cli download rinna/japanese-gpt-neox-small
```

### モデルが見つからない

```python
# キャッシュディレクトリを確認
chat_service = ChatService(
    model_name="rinna/japanese-gpt-neox-small",
    cache_dir="./models/chat_llm"  # 確認
)
```

## 📚 参考リンク

- [rinna/japanese-gpt-neox-small](https://huggingface.co/rinna/japanese-gpt-neox-small)
- [CyberAgent/open-calm-small](https://huggingface.co/cyberagent/open-calm-small)
- [Stability AI Japanese Models](https://huggingface.co/stabilityai)
- [OpenVINO Model Optimization](https://docs.openvino.ai/latest/index.html)
