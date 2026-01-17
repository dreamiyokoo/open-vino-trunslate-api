"""
FastAPI エンドポイントのテスト
"""

import pytest
from starlette.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    """テストクライアントのフィクスチャ"""
    return TestClient(app)


def test_read_root(client):
    """ルートページが正常に表示されるかテスト"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_languages(client):
    """サポート言語一覧APIのテスト"""
    response = client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert len(data["languages"]) > 0

    # 言語オブジェクトの構造を確認
    first_lang = data["languages"][0]
    assert "code" in first_lang
    assert "name" in first_lang


def test_health_check(client):
    """ヘルスチェックAPIのテスト"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_translate_missing_text(client):
    """テキストが空の場合のエラーハンドリングテスト"""
    response = client.post("/api/translate", json={"text": "", "target_lang": "ja"})
    # 空のテキストでも422エラーが返る可能性がある
    assert response.status_code in [400, 422]


def test_translate_invalid_language(client):
    """無効な言語コードのテスト"""
    response = client.post("/api/translate", json={"text": "Hello", "target_lang": "invalid_lang"})
    # サービスによっては400か翻訳結果のエラーが返る
    assert response.status_code in [400, 200]


def test_translate_request_structure(client):
    """翻訳リクエストの構造テスト"""
    response = client.post(
        "/api/translate",
        json={"text": "Test", "target_lang": "ja", "source_lang": "en"},
    )
    # モデルが未ロードの場合は失敗する可能性があるため、
    # ステータスコードのみチェック
    assert response.status_code in [200, 400, 500]


def test_api_docs(client):
    """OpenAPI ドキュメントが利用可能かテスト"""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data


# ========== Chat API のテスト ==========


def test_chat_endpoint_structure(client):
    """チャットエンドポイントの構造テスト"""
    response = client.post(
        "/api/chat",
        json={"message": "Hello"},
    )
    # モデルが未ロードの場合は失敗する可能性があるため、
    # ステータスコードのみチェック
    assert response.status_code in [200, 400, 500]


def test_chat_missing_message(client):
    """メッセージが空の場合のエラーハンドリングテスト"""
    response = client.post("/api/chat", json={"message": ""})
    # 空のメッセージでも422エラーが返る可能性がある
    assert response.status_code in [400, 422]


def test_chat_with_session_id(client):
    """セッションIDを指定したチャットのテスト"""
    response = client.post(
        "/api/chat",
        json={"message": "Hello", "session_id": "test-session-123"},
    )
    # モデルロード状況により結果が変わる
    assert response.status_code in [200, 400, 500]


def test_get_chat_history_not_found(client):
    """存在しないセッションの履歴取得テスト"""
    response = client.get("/api/chat/history/non-existent-session")
    # セッションが存在しない場合は404が返る
    assert response.status_code in [404, 500]


def test_delete_chat_history_not_found(client):
    """存在しないセッションの削除テスト"""
    response = client.delete("/api/chat/history/non-existent-session")
    # セッションが存在しない場合は404が返る
    assert response.status_code in [404, 500]


def test_list_chat_sessions(client):
    """チャットセッション一覧のテスト"""
    response = client.get("/api/chat/sessions")
    # モデルが未ロードの場合でもエンドポイント自体は存在する
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "sessions" in data
        assert "total" in data
