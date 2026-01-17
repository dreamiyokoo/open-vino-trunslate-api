"""
LangChain互換アダプター（オプション機能）

このモジュールは、ChatServiceをLangChainのインターフェースで使用できるようにするための
オプショナルな機能を提供します。

使用方法:
    from langchain_adapter import OpenVINOChatLangChain

    # LangChain互換のチャットインターフェース
    chat = OpenVINOChatLangChain()
    response = chat("こんにちは")

注意:
    - このモジュールを使用する場合は、langchainパッケージのインストールが必要です
    - pip install langchain
"""

from typing import Optional, Dict, Any, List
import logging

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # LangChainが利用できない場合のダミークラス
    class LLM:
        pass


from chat_service import ChatService

logger = logging.getLogger(__name__)


class OpenVINOChatLangChain(LLM):
    """
    LangChain互換のOpenVINOチャットラッパー

    ChatServiceをLangChainのLLMインターフェースでラップし、
    LangChainのエコシステムで利用可能にします。

    注意: 各インスタンスは独自のChatServiceを作成します。
    メモリを節約するには、既存のChatServiceインスタンスを
    chat_service引数で渡してください。

    例:
        >>> from langchain_adapter import OpenVINOChatLangChain
        >>> # 新しいモデルインスタンスを作成
        >>> chat = OpenVINOChatLangChain()
        >>> response = chat("こんにちは")
        >>> print(response)

        >>> # 既存のChatServiceを共有
        >>> from chat_service import ChatService
        >>> shared_service = ChatService()
        >>> chat1 = OpenVINOChatLangChain(chat_service=shared_service)
        >>> chat2 = OpenVINOChatLangChain(chat_service=shared_service)
    """

    chat_service: Optional[ChatService] = None
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(self, chat_service: Optional[ChatService] = None, **kwargs):
        """
        初期化

        Args:
            chat_service: 既存のChatServiceインスタンス（オプション）
                         指定しない場合は新しいインスタンスを作成
            model_name: 使用するLLMモデル名（chat_service未指定時のみ使用）
            session_id: セッションID（指定しない場合は新規作成）
            system_prompt: システムプロンプト
        """
        super().__init__(**kwargs)

        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChainが利用できません。以下のコマンドでインストールしてください:\n" "pip install langchain")

        # ChatServiceのインスタンスを取得または作成
        if chat_service is not None:
            self.chat_service = chat_service
            logger.info("Using shared ChatService instance")
        else:
            self.chat_service = ChatService(model_name=self.model_name)
            logger.info(f"Created new ChatService instance with model: {self.model_name}")

    @property
    def _llm_type(self) -> str:
        """LLMのタイプを返す"""
        return "openvino-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        プロンプトに対する応答を生成

        Args:
            prompt: 入力プロンプト
            stop: 停止トークン（現在未使用）
            run_manager: コールバックマネージャー
            **kwargs: 追加のキーワード引数

        Returns:
            生成されたテキスト
        """
        try:
            result = self.chat_service.chat(message=prompt, session_id=self.session_id, system_prompt=self.system_prompt)

            # セッションIDを保存
            if self.session_id is None:
                self.session_id = result["session_id"]

            return result["response"]

        except Exception as e:
            logger.error(f"LangChain adapter error: {e}")
            raise

    def get_history(self) -> Dict:
        """
        現在のセッションの会話履歴を取得

        Returns:
            会話履歴を含む辞書
        """
        if self.session_id is None:
            return {"messages": [], "session_id": None}

        return self.chat_service.get_history(self.session_id)

    def clear_history(self) -> None:
        """現在のセッションの会話履歴をクリア"""
        if self.session_id:
            self.chat_service.delete_history(self.session_id)
            self.session_id = None


# LangChainが利用できない場合のヘルパー関数
def is_langchain_available() -> bool:
    """
    LangChainが利用可能かどうかを確認

    Returns:
        LangChainが利用可能な場合はTrue
    """
    return LANGCHAIN_AVAILABLE


def create_langchain_chat(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    system_prompt: Optional[str] = None,
    chat_service: Optional[ChatService] = None,
) -> Any:
    """
    LangChain互換のチャットインスタンスを作成

    Args:
        model_name: 使用するLLMモデル名（chat_service未指定時のみ使用）
        system_prompt: システムプロンプト
        chat_service: 既存のChatServiceインスタンス（オプション）
                     指定するとメモリを節約できます

    Returns:
        OpenVINOChatLangChainインスタンス

    Raises:
        ImportError: LangChainが利用できない場合
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChainが利用できません。以下のコマンドでインストールしてください:\n" "pip install langchain")

    return OpenVINOChatLangChain(chat_service=chat_service, model_name=model_name, system_prompt=system_prompt)


# 使用例
if __name__ == "__main__":
    if LANGCHAIN_AVAILABLE:
        print("LangChain使用例:")
        print("-" * 50)

        # LangChain互換のチャットを作成
        chat = create_langchain_chat(system_prompt="あなたは親切で役立つAIアシスタントです。")

        # チャットを実行
        response = chat("こんにちは！")
        print("User: こんにちは！")
        print(f"Assistant: {response}")

        # 会話を続ける
        response = chat("今日の天気は？")
        print("User: 今日の天気は？")
        print(f"Assistant: {response}")

        # 履歴を取得
        history = chat.get_history()
        print(f"\n会話履歴: {len(history['messages'])} メッセージ")

    else:
        print("LangChainが利用できません。")
        print("インストール: pip install langchain")
