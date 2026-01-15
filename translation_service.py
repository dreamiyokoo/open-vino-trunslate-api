"""
OpenVINOを使った翻訳サービス
Helsinki-NLP/opus-mt モデルを使用して多言語翻訳を実現
"""

from pathlib import Path
from typing import Optional
from optimum.intel import OVModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationService:
    """OpenVINOベースの翻訳サービス"""

    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.loaded_models = {}

    def _get_model_name(self, source_lang: str, target_lang: str) -> str:
        """言語ペアに基づいてモデル名を取得"""
        # マルチリンガル モデルを使用
        lang_pairs = {
            ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
            ("en", "ja"): "Helsinki-NLP/opus-mt-en-ja",
            ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
            ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
            ("en", "ko"): "Helsinki-NLP/opus-mt-en-ko",
            ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
            ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
            ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
            ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
            ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
            ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
            ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
            ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
            ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
        }

        # 直接のペアがない場合は、英語を経由
        if (source_lang, target_lang) in lang_pairs:
            return lang_pairs[(source_lang, target_lang)]

        return None

    def _load_model(self, model_name: str):
        """モデルをロードしてキャッシュ"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        try:
            logger.info(f"Loading model: {model_name}")
            model_path = self.cache_dir / model_name.replace("/", "_")

            # モデルが既にエクスポートされているか確認
            if not model_path.exists():
                logger.info("Exporting model to OpenVINO format...")
                # PyTorchモデルをOpenVINO形式にエクスポート
                model = OVModelForSeq2SeqLM.from_pretrained(
                    model_name, export=True, compile=True
                )
                model.save_pretrained(model_path)
            else:
                logger.info("Loading cached OpenVINO model...")
                model = OVModelForSeq2SeqLM.from_pretrained(model_path, compile=True)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            translator = pipeline(
                "translation", model=model, tokenizer=tokenizer, device="cpu"
            )

            self.loaded_models[model_name] = translator
            logger.info(f"Model loaded successfully: {model_name}")
            return translator

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def translate(
        self, text: str, target_lang: str, source_lang: Optional[str] = None
    ) -> dict:
        """
        テキストを翻訳

        Args:
            text: 翻訳するテキスト
            target_lang: ターゲット言語コード (例: "ja", "en", "zh")
            source_lang: ソース言語コード (オプション。指定しない場合は自動検出を試みる)

        Returns:
            翻訳結果を含む辞書
        """
        try:
            # ソース言語が指定されていない場合は英語と仮定
            if source_lang is None:
                source_lang = self._detect_language(text)

            # 同じ言語の場合はそのまま返す
            if source_lang == target_lang:
                return {
                    "translated_text": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "original_text": text,
                }

            model_name = self._get_model_name(source_lang, target_lang)

            if model_name is None:
                # 英語を経由して翻訳
                if source_lang != "en" and target_lang != "en":
                    # source -> en -> target
                    intermediate_model = self._get_model_name(source_lang, "en")
                    final_model = self._get_model_name("en", target_lang)

                    if intermediate_model and final_model:
                        translator1 = self._load_model(intermediate_model)
                        english_text = translator1(text)[0]["translation_text"]

                        translator2 = self._load_model(final_model)
                        translated = translator2(english_text)[0]["translation_text"]

                        return {
                            "translated_text": translated,
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "original_text": text,
                            "via_english": True,
                        }

                return {
                    "error": f"Translation from {source_lang} to {target_lang} is not supported",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                }

            translator = self._load_model(model_name)
            result = translator(text)
            translated_text = result[0]["translation_text"]

            return {
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "original_text": text,
            }

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {"error": str(e), "original_text": text}

    def _detect_language(self, text: str) -> str:
        """
        簡易的な言語検出
        実際のプロダクションでは、より高度な言語検出ライブラリを使用することを推奨
        """
        # 日本語文字を含む場合
        if any(
            "\u3040" <= char <= "\u309f"
            or "\u30a0" <= char <= "\u30ff"
            or "\u4e00" <= char <= "\u9fff"
            for char in text
        ):
            return "ja"
        # 中国語文字を含む場合
        elif any("\u4e00" <= char <= "\u9fff" for char in text):
            return "zh"
        # 韓国語文字を含む場合
        elif any("\uac00" <= char <= "\ud7a3" for char in text):
            return "ko"
        # デフォルトは英語
        else:
            return "en"

    def get_supported_languages(self) -> list:
        """サポートされている言語のリストを返す"""
        return [
            {"code": "en", "name": "English"},
            {"code": "ja", "name": "Japanese (日本語)"},
            {"code": "zh", "name": "Chinese (中文)"},
            {"code": "ko", "name": "Korean (한국어)"},
            {"code": "fr", "name": "French (Français)"},
            {"code": "de", "name": "German (Deutsch)"},
            {"code": "es", "name": "Spanish (Español)"},
            {"code": "ru", "name": "Russian (Русский)"},
        ]
