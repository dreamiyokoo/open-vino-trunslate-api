"""
OpenVINOを使ったチャットサービス
軽量LLMを使用してチャット機能を実現
"""

from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging
import uuid
import threading

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    """OpenVINOベースのチャットサービス"""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir: str = "./models/chat_llm",
        max_history_messages: int = 20,
        session_timeout_minutes: int = 60,
        max_sessions: int = 100,
    ):
        """
        チャットサービスの初期化

        Args:
            model_name: 使用するLLMモデル名
            cache_dir: モデルキャッシュディレクトリ
            max_history_messages: セッションごとの最大メッセージ履歴数
            session_timeout_minutes: セッションタイムアウト（分）
            max_sessions: 最大セッション数
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_messages = max_history_messages
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions

        # モデルとトークナイザー
        self.model = None
        self.tokenizer = None

        # セッション管理（メモリ内）
        # session_id -> {"messages": [...], "system_prompt": str, "created_at": datetime, "last_access": datetime}
        self.sessions: Dict[str, Dict] = {}
        self.sessions_lock = threading.Lock()

        # モデルをロード
        self._load_model()

    def _load_model(self):
        """LLMモデルをロード"""
        try:
            logger.info(f"Loading chat model: {self.model_name}")
            model_path = self.cache_dir / self.model_name.replace("/", "_")

            # モデルが既にエクスポートされているか確認
            if not model_path.exists():
                logger.info("Exporting chat model to OpenVINO format...")
                self.model = OVModelForCausalLM.from_pretrained(self.model_name, export=True, compile=True)
                self.model.save_pretrained(model_path)
            else:
                logger.info("Loading cached OpenVINO chat model...")
                self.model = OVModelForCausalLM.from_pretrained(model_path, compile=True)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Chat model loaded successfully: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading chat model {self.model_name}: {e}")
            raise

    def _format_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        チャット履歴をプロンプト形式にフォーマット

        Args:
            messages: メッセージ履歴のリスト
            system_prompt: システムプロンプト

        Returns:
            フォーマットされたプロンプト文字列
        """
        # TinyLlamaのチャット形式に従う
        # 他のモデルを使用する場合は、このフォーマットを調整する必要がある
        formatted_messages = []

        if system_prompt:
            formatted_messages.append(f"<|system|>\n{system_prompt}</s>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                formatted_messages.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted_messages.append(f"<|assistant|>\n{content}</s>")

        # 最後にアシスタントの応答を促すプレフィックスを追加
        formatted_messages.append("<|assistant|>\n")

        return "\n".join(formatted_messages)

    def _generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        LLMを使用して応答を生成

        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数

        Returns:
            生成されたテキスト
        """
        try:
            # トークナイズ
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

            # 生成
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # デコード（入力部分を除外）
            generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _cleanup_old_sessions(self):
        """古いセッションをクリーンアップ"""
        with self.sessions_lock:
            current_time = datetime.now()
            expired_sessions = []

            for session_id, session in self.sessions.items():
                last_access = session.get("last_access", session["created_at"])
                if current_time - last_access > self.session_timeout:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")

            # セッション数が最大値を超えている場合、最も古いセッションを削除
            if len(self.sessions) > self.max_sessions:
                # 最終アクセス時刻でソート
                sorted_sessions = sorted(self.sessions.items(), key=lambda x: x[1].get("last_access", x[1]["created_at"]))
                # 古いセッションを削除
                num_to_remove = len(self.sessions) - self.max_sessions
                for i in range(num_to_remove):
                    session_id = sorted_sessions[i][0]
                    del self.sessions[session_id]
                    logger.info(f"Removed old session due to limit: {session_id}")

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        チャットメッセージを処理して応答を生成

        Args:
            message: ユーザーメッセージ
            session_id: セッションID（指定しない場合は新規作成）
            system_prompt: システムプロンプト

        Returns:
            応答情報を含む辞書
        """
        try:
            # 古いセッションをクリーンアップ
            self._cleanup_old_sessions()

            with self.sessions_lock:
                # セッションIDの処理
                if session_id is None or session_id not in self.sessions:
                    session_id = str(uuid.uuid4())
                    self.sessions[session_id] = {
                        "messages": [],
                        "system_prompt": system_prompt or "You are a helpful assistant.",
                        "created_at": datetime.now(),
                        "last_access": datetime.now(),
                    }
                elif system_prompt:
                    # 既存セッションのシステムプロンプトを更新
                    self.sessions[session_id]["system_prompt"] = system_prompt

                # 最終アクセス時刻を更新
                self.sessions[session_id]["last_access"] = datetime.now()

                session = self.sessions[session_id]

                # ユーザーメッセージを追加
                user_message = {
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now().isoformat(),
                }
                session["messages"].append(user_message)

                # メッセージ履歴を制限
                if len(session["messages"]) > self.max_history_messages * 2:
                    # 古いメッセージを削除（ペアで削除して会話の整合性を保つ）
                    session["messages"] = session["messages"][-(self.max_history_messages * 2) :]

                # プロンプトをフォーマット（ロック外で実行）
                prompt = self._format_prompt(session["messages"][:], session["system_prompt"])  # コピーを作成

            # 応答を生成（ロック外で実行 - 時間がかかる処理）
            response_text = self._generate_response(prompt)

            with self.sessions_lock:
                # アシスタントの応答を履歴に追加
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                }
                session["messages"].append(assistant_message)

            return {
                "response": response_text,
                "session_id": session_id,
                "timestamp": assistant_message["timestamp"],
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {"error": str(e)}

    def get_history(self, session_id: str) -> Dict:
        """
        セッションの会話履歴を取得

        Args:
            session_id: セッションID

        Returns:
            履歴情報を含む辞書
        """
        with self.sessions_lock:
            if session_id not in self.sessions:
                return {"error": "Session not found", "session_id": session_id}

            session = self.sessions[session_id]
            # 最終アクセス時刻を更新
            session["last_access"] = datetime.now()

            return {
                "session_id": session_id,
                "messages": session["messages"][:],  # コピーを返す
                "system_prompt": session["system_prompt"],
                "created_at": session["created_at"].isoformat(),
            }

    def delete_history(self, session_id: str) -> Dict:
        """
        セッションの会話履歴を削除

        Args:
            session_id: セッションID

        Returns:
            削除結果を含む辞書
        """
        with self.sessions_lock:
            if session_id not in self.sessions:
                return {"error": "Session not found", "session_id": session_id}

            del self.sessions[session_id]
            return {"success": True, "session_id": session_id, "message": "History deleted"}

    def list_sessions(self) -> Dict:
        """
        すべてのアクティブセッションをリスト

        Returns:
            セッションリスト
        """
        with self.sessions_lock:
            sessions_info = []
            # 辞書のコピーを作成してから反復
            sessions_copy = dict(self.sessions)

            for sid, session in sessions_copy.items():
                sessions_info.append(
                    {
                        "session_id": sid,
                        "message_count": len(session["messages"]),
                        "created_at": session["created_at"].isoformat(),
                        "last_access": session.get("last_access", session["created_at"]).isoformat(),
                    }
                )

            return {"sessions": sessions_info, "total": len(sessions_info)}
