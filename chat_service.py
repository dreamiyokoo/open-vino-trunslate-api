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
    """OpenVINOベースのチャットサービス（複数モデル対応）"""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir: str = "./models/chat_llm",
        max_history_messages: int = 20,
        session_timeout_minutes: int = 60,
        max_sessions: int = 100,
        use_mock: bool = False,  # モックモードを追加
    ):
        """
        チャットサービスの初期化

        Args:
            model_name: 使用するLLMモデル名
            cache_dir: モデルキャッシュディレクトリ
            max_history_messages: セッションごとの最大メッセージ履歴数
            session_timeout_minutes: セッションタイムアウト（分）
            max_sessions: 最大セッション数
            use_mock: モックモードを使用（開発・デモ用）
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_messages = max_history_messages
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions
        self.use_mock = use_mock

        # モデルとトークナイザー（複数モデルをキャッシュ）
        self.models: Dict[str, any] = {}  # model_name -> model
        self.tokenizers: Dict[str, any] = {}  # model_name -> tokenizer
        self.models_lock = threading.Lock()

        # セッション管理（メモリ内）
        # session_id -> {"messages": [...], "system_prompt": str, "model_name": str, "created_at": datetime, "last_access": datetime}
        self.sessions: Dict[str, Dict] = {}
        self.sessions_lock = threading.Lock()

        # デフォルトモデルをロード（モックモードでない場合のみ）
        if not self.use_mock:
            self._load_model(self.model_name)
        else:
            logger.info("Chat service initialized in MOCK mode")

    def _load_model(self, model_name: str = None):
        """LLMモデルをロード"""
        if model_name is None:
            model_name = self.model_name

        # 既にロード済みの場合はスキップ
        with self.models_lock:
            if model_name in self.models:
                logger.info(f"Model already loaded: {model_name}")
                return

        try:
            logger.info(f"Loading chat model: {model_name}")
            model_path = self.cache_dir / model_name.replace("/", "_")

            # モデルが既にエクスポートされているか確認
            if not model_path.exists():
                logger.info("Exporting chat model to OpenVINO format...")
                model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=True)
                model.save_pretrained(model_path)
            else:
                logger.info("Loading cached OpenVINO chat model...")
                model = OVModelForCausalLM.from_pretrained(model_path, compile=True)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # パディングトークンの設定
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            with self.models_lock:
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer

            logger.info(f"Chat model loaded successfully: {model_name}")

        except Exception as e:
            logger.error(f"Error loading chat model {model_name}: {e}")
            raise

    def get_loaded_models(self) -> List[str]:
        """ロード済みのモデル一覧を取得"""
        with self.models_lock:
            return list(self.models.keys())

    def _format_qwen_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Qwen2.5のチャット形式にフォーマット"""
        formatted_parts = []
        if system_prompt:
            formatted_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        formatted_parts.append("<|im_start|>assistant\n")
        return "\n".join(formatted_parts)

    def _format_tinyllama_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """TinyLlamaのチャット形式にフォーマット"""
        formatted_parts = []
        if system_prompt:
            formatted_parts.append(f"<|system|>\n{system_prompt}</s>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{content}</s>")

        formatted_parts.append("<|assistant|>")
        return "\n".join(formatted_parts)

    def _format_simple_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """日本語モデル用のシンプルな形式にフォーマット"""
        formatted_parts = []
        if system_prompt:
            formatted_parts.append(f"指示: {system_prompt}")

        # 会話履歴を追加（最後のいくつかのみ）
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"質問: {content}")
            elif role == "assistant":
                formatted_parts.append(f"回答: {content}")

        formatted_parts.append("回答:")
        return "\n".join(formatted_parts)

    def _format_prompt(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, model_name: str = None
    ) -> str:
        """
        チャット履歴をプロンプト形式にフォーマット

        Args:
            messages: メッセージ履歴のリスト
            system_prompt: システムプロンプト
            model_name: モデル名（フォーマット判定用）

        Returns:
            フォーマットされたプロンプト文字列
        """
        if model_name and "Qwen" in model_name:
            return self._format_qwen_prompt(messages, system_prompt)
        elif model_name and "TinyLlama" in model_name:
            return self._format_tinyllama_prompt(messages, system_prompt)
        else:
            return self._format_simple_prompt(messages, system_prompt)

    def _generate_response(self, prompt: str, model_name: str, max_new_tokens: int = 256) -> str:
        """
        LLMを使用して応答を生成

        Args:
            prompt: 入力プロンプト
            model_name: 使用するモデル名
            max_new_tokens: 生成する最大トークン数

        Returns:
            生成されたテキスト
        """
        # デバッグ用にプロンプトをログ出力
        logger.info(f"=== Prompt for model {model_name} ===\n{prompt}\n=== End Prompt ===")

        # モックモードの場合
        if self.use_mock:
            return self._generate_mock_response(prompt)

        # モデルがロードされていない場合はロード
        if model_name not in self.models:
            self._load_model(model_name)

        try:
            with self.models_lock:
                model = self.models[model_name]
                tokenizer = self.tokenizers[model_name]

            # トークナイズ
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

            # 生成パラメータの調整
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # デコード（入力部分を除外）
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # デバッグ: 生成された生テキストをログ出力
            logger.info(f"=== Raw generated text ===\n{generated_text}\n=== End Raw ===")

            # 特殊トークンやフォーマット記号を削除
            generated_text = (
                generated_text.replace("</s>", "")
                .replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .replace("<|assistant|>", "")
                .replace("<|user|>", "")
                .replace("質問:", "")
                .replace("回答:", "")
                .replace("指示:", "")
                .strip()
            )

            # 余分な生成を防ぐ - 次のターンが始まったら切り取る
            for delimiter in ["\n<|im_start|>", "\n質問:", "\n回答:", "\n指示:"]:
                if delimiter in generated_text:
                    generated_text = generated_text.split(delimiter)[0].strip()
                    break

            # 空の応答の場合のフォールバック
            if not generated_text:
                generated_text = "申し訳ありませんが、応答を生成できませんでした。"

            logger.info(f"=== Final response ===\n{generated_text}\n=== End Final ===")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _generate_mock_response(self, prompt: str) -> str:
        """
        モック応答を生成（開発・デモ用）

        Args:
            prompt: 入力プロンプト

        Returns:
            モック応答
        """
        # プロンプトから最後のユーザーメッセージを抽出
        user_message = ""
        if "<|user|>" in prompt:
            parts = prompt.split("<|user|>")
            if len(parts) > 1:
                last_user = parts[-1].split("</s>")[0].strip()
                user_message = last_user.lower()

        # シンプルなルールベースの応答
        if "こんにちは" in user_message or "hello" in user_message:
            return "こんにちは！どのようにお手伝いできますか？"
        elif "ありがとう" in user_message or "thank" in user_message:
            return "どういたしまして！他に何かお手伝いできることはありますか？"
        elif "さようなら" in user_message or "bye" in user_message:
            return "さようなら！また何かあればお気軽にどうぞ。"
        elif "天気" in user_message or "weather" in user_message:
            return "申し訳ありませんが、私は天気情報にアクセスできません。お近くの気象情報をご確認ください。"
        elif "名前" in user_message or "name" in user_message:
            return "私はOpenVINO Translation & Chat APIのアシスタントです。"
        elif "?" in user_message or "？" in user_message:
            return f"「{user_message[:50]}」についてのご質問ですね。申し訳ありませんが、現在はモックモードで動作しているため、詳細な回答は提供できません。"
        else:
            return "ご質問ありがとうございます。現在はモックモードで動作しているため、限定的な応答のみ提供しています。実際のLLMモデルを使用する場合は、適切な日本語対応モデル（rinna/japanese-gpt-neox-small等）の設定をご検討ください。"

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
        model_name: Optional[str] = None,
    ) -> Dict:
        """
        チャットメッセージを処理して応答を生成

        Args:
            message: ユーザーメッセージ
            session_id: セッションID（指定しない場合は新規作成）
            system_prompt: システムプロンプト
            model_name: 使用するモデル名（指定しない場合はデフォルト）

        Returns:
            応答情報を含む辞書
        """
        # モデル名のデフォルト設定
        if model_name is None:
            model_name = self.model_name

        try:
            # 古いセッションをクリーンアップ
            self._cleanup_old_sessions()

            with self.sessions_lock:
                # セッションIDの処理
                if session_id is None or session_id not in self.sessions:
                    session_id = str(uuid.uuid4())
                    # デフォルトのシステムプロンプトを日本語に対応
                    default_system_prompt = (
                        system_prompt
                        or "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
                    )
                    self.sessions[session_id] = {
                        "messages": [],
                        "system_prompt": default_system_prompt,
                        "model_name": model_name,  # セッションごとのモデルを記録
                        "created_at": datetime.now(),
                        "last_access": datetime.now(),
                    }
                elif system_prompt:
                    # 既存セッションのシステムプロンプトを更新
                    self.sessions[session_id]["system_prompt"] = system_prompt

                # モデルが指定された場合、セッションのモデルを更新
                if model_name:
                    self.sessions[session_id]["model_name"] = model_name

                # 最終アクセス時刻を更新
                self.sessions[session_id]["last_access"] = datetime.now()

                session = self.sessions[session_id]
                used_model = session["model_name"]

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
                prompt = self._format_prompt(session["messages"][:], session["system_prompt"], model_name=used_model)

            # 応答を生成（ロック外で実行 - 時間がかかる処理）
            response_text = self._generate_response(prompt, used_model)

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
                "model": used_model,  # 使用したモデルを返す
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
