"""
OpenVINO AI Toolkit - FastAPIアプリケーション
"""

import threading
import logging
import uvicorn
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from translation_service import TranslationService
from chat_service import ChatService
import config

logger = logging.getLogger(__name__)

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="OpenVINO AI Toolkit",
    description="OpenVINOを使った高速AI翻訳とチャット機能を提供するWebサービス",
    version="2.0.0",
)

# 静的ファイルとテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 翻訳サービスの初期化
translation_service = TranslationService()

# チャットサービスの初期化（遅延ロード、スレッドセーフ）

chat_service = None
chat_service_lock = threading.Lock()


def get_chat_service():
    """チャットサービスのインスタンスを取得（遅延ロード、スレッドセーフ）"""
    global chat_service

    if chat_service is None:
        with chat_service_lock:
            # ダブルチェックロッキングパターン
            if chat_service is None:
                # デフォルトモデルを config から取得
                default_model_key = config.DEFAULT_CHAT_MODEL
                model_name = config.get_model_name(default_model_key)

                chat_service = ChatService(model_name=model_name, use_mock=False, **config.CHAT_CONFIG)
                logger.info(f"Chat service initialized with model: {model_name}")

    return chat_service


# リクエストモデル
class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, world!",
                "target_lang": "ja",
                "source_lang": "en",
            }
        }


# レスポンスモデル
class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    original_text: str
    via_english: Optional[bool] = False


# Chat リクエストモデル
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    use_langchain: Optional[bool] = False
    translate_to: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "こんにちは",
                "session_id": "optional-session-id",
                "model": "tinyllama",
                "system_prompt": "あなたは親切なアシスタントです",
                "use_langchain": False,
                "translate_to": "en",
            }
        }


# Chat レスポンスモデル
class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    translated_response: Optional[str] = None


# Chat履歴レスポンスモデル
class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict]
    system_prompt: Optional[str] = None
    created_at: Optional[str] = None


# ルートページ - Webインターフェース
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """翻訳Webインターフェースを表示"""
    languages = translation_service.get_supported_languages()
    return templates.TemplateResponse("index.html", {"request": request, "languages": languages})


# 翻訳API エンドポイント
@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    テキストを翻訳するAPIエンドポイント

    - **text**: 翻訳するテキスト
    - **target_lang**: ターゲット言語コード (例: "ja", "en", "zh")
    - **source_lang**: ソース言語コード (オプション)
    """
    result = translation_service.translate(
        text=request.text,
        target_lang=request.target_lang,
        source_lang=request.source_lang,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return TranslationResponse(**result)


# サポートされている言語のリストを取得
@app.get("/api/languages")
async def get_languages():
    """
    サポートされている言語のリストを取得
    """
    return {"languages": translation_service.get_supported_languages()}


# ヘルスチェック
@app.get("/api/health")
async def health_check():
    """APIヘルスチェック"""
    return {"status": "healthy", "service": "OpenVINO Translation API"}


# ========== Chat API エンドポイント ==========


def _validate_chat_request(request: ChatRequest) -> None:
    """チャットリクエストの検証"""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if request.session_id and len(request.session_id) > 100:
        raise HTTPException(status_code=400, detail="Session ID is too long")


def _resolve_model_name(model_key: Optional[str]) -> Optional[str]:
    """モデルキーからHugging Face名を解決"""
    if not model_key:
        return None
    try:
        return config.get_model_name(model_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _translate_response(response_text: str, target_lang: str) -> Optional[str]:
    """応答を翻訳"""
    try:
        translation_result = translation_service.translate(
            text=response_text,
            target_lang=target_lang,
            source_lang=None,
        )
        if "error" not in translation_result:
            return translation_result["translated_text"]
        else:
            logger.warning(f"Translation failed: {translation_result['error']}")
    except Exception as e:
        logger.error(f"Translation error: {e}")
    return None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    チャットメッセージを送信して応答を取得

    - **message**: ユーザーメッセージ
    - **session_id**: セッションID (オプション、指定しない場合は新規作成)
    - **system_prompt**: システムプロンプト (オプション)
    - **model**: 使用するモデル (オプション、例: "tinyllama", "japanese-gpt-neox")
    - **translate_to**: 応答を翻訳する言語コード (オプション)
    """
    _validate_chat_request(request)
    service = get_chat_service()
    model_name = _resolve_model_name(request.model)

    result = service.chat(
        message=request.message.strip(),
        session_id=request.session_id,
        system_prompt=request.system_prompt,
        model_name=model_name,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    translated_response = None
    if request.translate_to:
        translated_response = _translate_response(result["response"], request.translate_to)

    return ChatResponse(
        response=result["response"],
        session_id=result["session_id"],
        timestamp=result["timestamp"],
        translated_response=translated_response,
    )


@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """
    セッションの会話履歴を取得

    - **session_id**: セッションID
    """
    service = get_chat_service()
    result = service.get_history(session_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return ChatHistoryResponse(**result)


@app.delete("/api/chat/history/{session_id}")
async def delete_chat_history(session_id: str):
    """
    セッションの会話履歴を削除

    - **session_id**: セッションID
    """
    service = get_chat_service()
    result = service.delete_history(session_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/api/chat/models")
async def get_available_models():
    """
    利用可能なチャットモデルの一覧を取得

    Ollamaライクなモデル管理機能
    """
    models = config.list_available_models()

    # 現在ロードされているモデルの情報を追加
    service = get_chat_service()
    loaded_models = service.get_loaded_models()

    for model in models:
        model["loaded"] = model["name"] in loaded_models

    return {
        "models": models,
        "default": config.DEFAULT_CHAT_MODEL,
        "loaded_count": len(loaded_models),
    }


@app.get("/api/chat/models/{model_key}")
async def get_model_info(model_key: str):
    """
    特定のモデルの詳細情報を取得

    - **model_key**: モデルのキー (例: "tinyllama", "japanese-gpt-neox")
    """
    try:
        info = config.get_model_info(model_key)
        service = get_chat_service()
        loaded = info["name"] in service.get_loaded_models()

        return {
            "key": model_key,
            **info,
            "loaded": loaded,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/chat/sessions")
async def list_chat_sessions():
    """
    すべてのアクティブなチャットセッションをリスト
    """
    service = get_chat_service()
    return service.list_sessions()


if __name__ == "__main__":
    # アプリケーションを起動
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
