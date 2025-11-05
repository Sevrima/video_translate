"""
Language Detection Module.

Detects the source language of text using fasttext or fallback methods.
Returns language codes compatible with NLLB-200 format (e.g., "eng_Latn", "deu_Latn").
"""
import os
from pathlib import Path


# NLLB-200 language code mapping
# Maps common language codes to NLLB format
NLLB_LANGUAGE_MAP = {
    'en': 'eng_Latn',
    'de': 'deu_Latn',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
    'ar': 'arb_Arab',
    'hi': 'hin_Deva',
    'tr': 'tur_Latn',
    'pl': 'pol_Latn',
    'nl': 'nld_Latn',
    'sv': 'swe_Latn',
    'da': 'dan_Latn',
    'fi': 'fin_Latn',
    'no': 'nob_Latn',
    'cs': 'ces_Latn',
    'el': 'ell_Grek',
    'he': 'heb_Hebr',
    'th': 'tha_Thai',
    'vi': 'vie_Latn',
    'id': 'ind_Latn',
    'ms': 'zsm_Latn',
    'tl': 'tgl_Latn',
    'sw': 'swh_Latn',
    'uk': 'ukr_Cyrl',
    'ro': 'ron_Latn',
    'hu': 'hun_Latn',
    'bg': 'bul_Cyrl',
    'sk': 'slk_Latn',
    'lt': 'lit_Latn',
    'lv': 'lvs_Latn',
    'et': 'est_Latn',
    'sl': 'slv_Latn',
    'hr': 'hrv_Latn',
    'sr': 'srp_Cyrl',
    'ca': 'cat_Latn',
    'gl': 'glg_Latn',
    'eu': 'eus_Latn',
}


class LanguageDetector:
    """Detect language of text using multiple methods."""

    def __init__(self, method='fasttext'):
        """
        Initialize language detector.

        Args:
            method (str): Detection method - 'fasttext' or 'langdetect'
        """
        self.method = method
        self.fasttext_model = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the language detection model."""
        if self.method == 'fasttext':
            # Using HuggingFace model - will be loaded on first use
            self.fasttext_model = None
            print("✓ Language detector initialized (will load from HuggingFace on first use)")
        else:
            print("⚠ Using langdetect method")

    def _load_fasttext_model(self):
        """Load fasttext language identification model from HuggingFace."""
        try:
            from transformers import pipeline

            if self.fasttext_model is None:
                print("▶ Loading language detection model from HuggingFace...")
                self.fasttext_model = pipeline(
                    "text-classification",
                    model="facebook/fasttext-language-identification",
                    top_k=1
                )
                print("✓ Language detection model loaded")

            return True

        except Exception as e:
            print(f"⚠ Could not load fasttext model from HuggingFace: {e}")
            return False

    def detect(self, text, k=1):
        """
        Detect language of text.

        Args:
            text (str): Text to detect language from
            k (int): Number of top predictions to return

        Returns:
            str: NLLB language code (e.g., "eng_Latn")
        """
        if not text or not text.strip():
            raise ValueError("Cannot detect language of empty text")

        # Clean text for detection
        text_sample = text.strip()[:1000]  # Use first 1000 chars for detection

        if self.method == 'fasttext':
            return self._detect_fasttext(text_sample, k)
        else:
            return self._detect_langdetect(text_sample)

    def _detect_fasttext(self, text, k=1):
        """
        Detect language using fasttext from HuggingFace.

        Args:
            text (str): Text to analyze
            k (int): Number of predictions

        Returns:
            str: NLLB language code
        """
        # Load model if not loaded
        if self.fasttext_model is None:
            if not self._load_fasttext_model():
                # Fallback to langdetect
                print("⚠ Falling back to langdetect method")
                self.method = 'langdetect'
                return self._detect_langdetect(text)

        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ').strip()

        # Truncate if too long (model has limits)
        if len(text) > 512:
            text = text[:512]

        # Predict language
        predictions = self.fasttext_model(text)

        # Extract language code (HuggingFace returns format like '__label__eng_Latn')
        detected_label = predictions[0]['label']
        confidence = predictions[0]['score']

        # Remove __label__ prefix if present
        detected_lang = detected_label.replace('__label__', '')

        print(f"  Detected language: {detected_lang} (confidence: {confidence:.2f})")

        # If already in NLLB format (e.g., 'eng_Latn'), return as-is
        if '_' in detected_lang:
            return detected_lang

        # Otherwise, convert from 2-letter code to NLLB format
        nllb_code = NLLB_LANGUAGE_MAP.get(detected_lang, 'eng_Latn')
        return nllb_code

    def _detect_langdetect(self, text):
        """
        Detect language using langdetect library (fallback).

        Args:
            text (str): Text to analyze

        Returns:
            str: NLLB language code
        """
        try:
            from langdetect import detect, detect_langs

            detected = detect(text)
            print(f"  Detected language: {detected}")

            # Convert to NLLB format
            nllb_code = NLLB_LANGUAGE_MAP.get(detected, 'eng_Latn')
            return nllb_code

        except Exception as e:
            print(f"⚠ Language detection failed: {e}")
            print("  Defaulting to English (eng_Latn)")
            return 'eng_Latn'

    def detect_from_texts(self, texts):
        """
        Detect language from multiple text samples.

        Uses the first non-empty text or combines multiple texts.

        Args:
            texts (list): List of text strings

        Returns:
            str: NLLB language code
        """
        # Filter out empty texts
        valid_texts = [t.strip() for t in texts if t and t.strip()]

        if not valid_texts:
            raise ValueError("No valid text to detect language from")

        # Use first few texts for detection (combine them)
        sample_texts = valid_texts[:5]  # Use first 5 segments
        combined_text = ' '.join(sample_texts)

        return self.detect(combined_text)


def detect_language(text, method='fasttext'):
    """
    Convenience function to detect language from text.

    Args:
        text (str): Text to analyze
        method (str): Detection method

    Returns:
        str: NLLB language code
    """
    detector = LanguageDetector(method=method)
    return detector.detect(text)


# Example usage
if __name__ == "__main__":
    # Test language detection
    test_texts = {
        'English': "Hello, how are you today? This is a test of language detection.",
        'German': "Hallo, wie geht es dir heute? Dies ist ein Test der Spracherkennung.",
        'Spanish': "Hola, ¿cómo estás hoy? Esta es una prueba de detección de idioma.",
        'French': "Bonjour, comment allez-vous aujourd'hui? C'est un test de détection de langue."
    }

    detector = LanguageDetector(method='fasttext')

    print("Language Detection Test:\n")
    for expected_lang, text in test_texts.items():
        print(f"Expected: {expected_lang}")
        print(f"Text: {text[:50]}...")
        detected = detector.detect(text)
        print(f"Detected NLLB code: {detected}\n")
