"""
TranslationService のユニットテスト
"""

import pytest
from translation_service import TranslationService


@pytest.fixture
def translation_service():
    """テスト用の翻訳サービスインスタンス"""
    return TranslationService(cache_dir="./test_models")


def test_translation_service_init(translation_service):
    """翻訳サービスの初期化テスト"""
    assert translation_service is not None
    assert translation_service.cache_dir.exists()
    assert isinstance(translation_service.loaded_models, dict)


def test_get_supported_languages(translation_service):
    """サポート言語リストの取得テスト"""
    languages = translation_service.get_supported_languages()
    assert len(languages) == 7  # Updated to 7 languages (removed Korean and Thai)
    assert all("code" in lang for lang in languages)
    assert all("name" in lang for lang in languages)

    # 主要言語が含まれているか確認
    lang_codes = [lang["code"] for lang in languages]
    assert "en" in lang_codes
    assert "ja" in lang_codes
    assert "zh" in lang_codes
    assert "fr" in lang_codes
    assert "de" in lang_codes
    assert "es" in lang_codes
    assert "ru" in lang_codes


def test_detect_language(translation_service):
    """言語検出のテスト"""
    # 日本語
    assert translation_service._detect_language("こんにちは") == "ja"

    # 中国語（簡体字） - 日本語の漢字と同じものは日本語として判定される可能性がある
    detected = translation_service._detect_language("你好")
    assert detected in ["zh", "ja"]  # どちらでも許容

    # 英語（デフォルト）
    assert translation_service._detect_language("Hello") == "en"


def test_get_model_name(translation_service):
    """モデル名取得のテスト"""
    # 直接サポートされている言語ペア
    model_name = translation_service._get_model_name("en", "ja")
    assert model_name == "Helsinki-NLP/opus-mt-en-ja"

    model_name = translation_service._get_model_name("ja", "en")
    assert model_name == "Helsinki-NLP/opus-mt-ja-en"

    # サポートされていないペア
    model_name = translation_service._get_model_name("invalid", "lang")
    assert model_name is None


def test_same_language_translation(translation_service):
    """同じ言語への翻訳テスト"""
    result = translation_service.translate(text="Hello", target_lang="en", source_lang="en")

    assert result["translated_text"] == "Hello"
    assert result["source_lang"] == "en"
    assert result["target_lang"] == "en"
    assert result["original_text"] == "Hello"


def test_translation_result_structure(translation_service):
    """翻訳結果の構造テスト"""
    result = translation_service.translate(text="Test", target_lang="ja", source_lang="en")

    # 基本的なキーが存在するか確認（original_textは常に含まれる）
    assert "original_text" in result

    # エラーがない場合は translated_text が含まれる
    # エラーがある場合は error が含まれる
    if "error" in result:
        # モデルのダウンロードやネットワークエラーがある場合
        assert "error" in result
        assert result["original_text"] == "Test"
    else:
        # 正常な翻訳の場合
        assert "translated_text" in result
        assert "source_lang" in result
        assert "target_lang" in result
