"""
チャットサービスのテスト
"""

from unittest.mock import patch, MagicMock
from chat_service import ChatService


class TestChatService:
    """ChatServiceクラスのテスト"""

    @patch("chat_service.OVModelForCausalLM")
    @patch("chat_service.AutoTokenizer")
    def test_chat_service_initialization(self, mock_tokenizer, mock_model):
        """チャットサービスの初期化テスト"""

        # モックの設定
        mock_model.from_pretrained = MagicMock()
        mock_tokenizer.from_pretrained = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        service = ChatService()

        assert service.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert service.sessions == {}

    @patch("chat_service.OVModelForCausalLM")
    @patch("chat_service.AutoTokenizer")
    def test_chat_creates_new_session(self, mock_tokenizer, mock_model):
        """新規セッションの作成テスト"""

        # モックの設定
        mock_model_instance = MagicMock()
        mock_model_instance.generate = MagicMock(return_value=[[1, 2, 3, 4, 5]])  # ダミートークン
        mock_model.from_pretrained = MagicMock(return_value=mock_model_instance)

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer_instance.return_value = {"input_ids": MagicMock(shape=(1, 3))}
        mock_tokenizer_instance.decode = MagicMock(return_value="こんにちは！")
        mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer_instance)

        service = ChatService()
        service.tokenizer = mock_tokenizer_instance
        service.model = mock_model_instance

        result = service.chat("こんにちは")

        assert "response" in result
        assert "session_id" in result
        assert "timestamp" in result
        assert result["session_id"] in service.sessions

    @patch("chat_service.OVModelForCausalLM")
    @patch("chat_service.AutoTokenizer")
    def test_chat_reuses_existing_session(self, mock_tokenizer, mock_model):
        """既存セッションの再利用テスト"""

        # モックの設定
        mock_model_instance = MagicMock()
        mock_model_instance.generate = MagicMock(return_value=[[1, 2, 3, 4, 5]])  # ダミートークン
        mock_model.from_pretrained = MagicMock(return_value=mock_model_instance)

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer_instance.return_value = {"input_ids": MagicMock(shape=(1, 3))}
        mock_tokenizer_instance.decode = MagicMock(return_value="応答です")
        mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer_instance)

        service = ChatService()
        service.tokenizer = mock_tokenizer_instance
        service.model = mock_model_instance

        # 最初のメッセージ
        result1 = service.chat("こんにちは")
        session_id = result1["session_id"]

        # 同じセッションで2番目のメッセージ
        result2 = service.chat("元気ですか？", session_id=session_id)

        assert result2["session_id"] == session_id
        assert len(service.sessions[session_id]["messages"]) == 4  # 2往復

    @patch("chat_service.OVModelForCausalLM")
    @patch("chat_service.AutoTokenizer")
    def test_get_history(self, mock_tokenizer, mock_model):
        """履歴取得のテスト"""

        # モックの設定
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_model.from_pretrained = MagicMock(return_value=mock_model_instance)
        mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer_instance)

        service = ChatService()

        # テスト用セッションを手動で作成
        from datetime import datetime

        test_session_id = "test-session-123"
        service.sessions[test_session_id] = {
            "messages": [
                {"role": "user", "content": "こんにちは", "timestamp": "2024-01-01"},
                {
                    "role": "assistant",
                    "content": "こんにちは！",
                    "timestamp": "2024-01-01",
                },
            ],
            "system_prompt": "あなたは親切なアシスタントです",
            "created_at": datetime.now(),
        }

        result = service.get_history(test_session_id)

        assert result["session_id"] == test_session_id
        assert len(result["messages"]) == 2
        assert "system_prompt" in result

    @patch("chat_service.OVModelForCausalLM")
    @patch("chat_service.AutoTokenizer")
    def test_get_history_not_found(self, mock_tokenizer, mock_model):
        """存在しないセッションの履歴取得テスト"""

        # モックの設定
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_model.from_pretrained = MagicMock(return_value=mock_model_instance)
        mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer_instance)

        service = ChatService()
        result = service.get_history("non-existent-session")

        assert "error" in result
        assert result["error"] == "Session not found"

    @patch("chat_service.OVModelForCausalLM")
    @patch("chat_service.AutoTokenizer")
    def test_delete_history(self, mock_tokenizer, mock_model):
        """履歴削除のテスト"""

        # モックの設定
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_model.from_pretrained = MagicMock(return_value=mock_model_instance)
        mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer_instance)

        service = ChatService()

        # テスト用セッションを作成
        from datetime import datetime

        test_session_id = "test-session-123"
        service.sessions[test_session_id] = {
            "messages": [],
            "system_prompt": "test",
            "created_at": datetime.now(),
        }

        result = service.delete_history(test_session_id)

        assert result["success"] is True
        assert test_session_id not in service.sessions

    @patch("chat_service.OVModelForCausalLM")
    @patch("chat_service.AutoTokenizer")
    def test_list_sessions(self, mock_tokenizer, mock_model):
        """セッション一覧のテスト"""

        # モックの設定
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_model.from_pretrained = MagicMock(return_value=mock_model_instance)
        mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer_instance)

        service = ChatService()

        # テスト用セッションを作成
        from datetime import datetime

        service.sessions["session1"] = {
            "messages": [{"role": "user", "content": "test"}],
            "system_prompt": "test",
            "created_at": datetime.now(),
        }
        service.sessions["session2"] = {
            "messages": [],
            "system_prompt": "test",
            "created_at": datetime.now(),
        }

        result = service.list_sessions()

        assert result["total"] == 2
        assert len(result["sessions"]) == 2
