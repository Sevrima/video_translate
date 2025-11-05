"""
Text Translation Module using NLLB-200-3.3B.

Translates text from source language to target language using Meta's
No Language Left Behind (NLLB) 3.3B parameter model via HuggingFace.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


class TextTranslator:
    """Translate text using NLLB-200-3.3B model."""

    def __init__(self, model_name="facebook/nllb-200-3.3B", device=None):
        """
        Initialize text translator.

        Args:
            model_name (str): HuggingFace model name
            device (str): Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

        print(f"Initializing translator with {model_name}")
        print(f"Device: {self.device}")

        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def load_model(self):
        """
        Load translation model and tokenizer.

        This will download the model (~6.6GB) on first run.
        """
        if self.model is not None:
            print("Model already loaded")
            return

        print(f"\nLoading model: {self.model_name}")
        print("This may take a few minutes on first run...")

        try:
            # Load tokenizer
            print("▶ Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("  ✓ Tokenizer loaded")

            # Load model
            print("▶ Loading model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("  ✓ Model loaded and moved to device")

            print(f"\n✓ Model ready for translation on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def translate(self, text, src_lang, tgt_lang, max_length=512, num_beams=5):
        """
        Translate a single text.

        Args:
            text (str): Text to translate
            src_lang (str): Source language code (NLLB format, e.g., "eng_Latn")
            tgt_lang (str): Target language code (NLLB format, e.g., "deu_Latn")
            max_length (int): Maximum length of generated translation
            num_beams (int): Number of beams for beam search

        Returns:
            str: Translated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not text or not text.strip():
            return text  # Return empty text as-is

        try:
            # Set source language for tokenizer
            self.tokenizer.src_lang = src_lang

            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Get target language token ID
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )

            # Decode translation
            translation = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            return translation

        except Exception as e:
            print(f"⚠ Translation error for text: {text[:50]}... - {e}")
            return text  # Return original text on error

    def translate_batch(self, texts, src_lang, tgt_lang, batch_size=8, max_length=512):
        """
        Translate multiple texts in batches.

        Args:
            texts (list): List of texts to translate
            src_lang (str): Source language code
            tgt_lang (str): Target language code
            batch_size (int): Number of texts to translate at once
            max_length (int): Maximum length per translation

        Returns:
            list: List of translated texts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"\nTranslating {len(texts)} texts from {src_lang} to {tgt_lang}")
        print(f"Batch size: {batch_size}")

        translations = []

        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]

            # Filter out empty texts
            non_empty_indices = [j for j, t in enumerate(batch) if t and t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]

            if not non_empty_texts:
                # All texts in batch are empty
                translations.extend(batch)
                continue

            try:
                # Set source language
                self.tokenizer.src_lang = src_lang

                # Tokenize batch
                inputs = self.tokenizer(
                    non_empty_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)

                # Get target language token ID
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

                # Generate translations
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=max_length,
                        num_beams=5,
                        early_stopping=True
                    )

                # Decode translations
                batch_translations = self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )

                # Reconstruct full batch with empty texts preserved
                result_batch = batch.copy()
                for j, trans in zip(non_empty_indices, batch_translations):
                    result_batch[j] = trans

                translations.extend(result_batch)

            except Exception as e:
                print(f"⚠ Batch translation error: {e}")
                # Return original texts on error
                translations.extend(batch)

        print(f"✓ Translation complete: {len(translations)} texts")
        return translations

    def unload_model(self):
        """Unload model from memory to free up resources."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if self.device == 'cuda':
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print("✓ Model unloaded from memory")


def translate_texts(texts, src_lang, tgt_lang, model_name="facebook/nllb-200-3.3B", device=None):
    """
    Convenience function to translate texts.

    Args:
        texts (list): List of texts to translate
        src_lang (str): Source language code (NLLB format)
        tgt_lang (str): Target language code (NLLB format)
        model_name (str): HuggingFace model name
        device (str): Device to use

    Returns:
        list: Translated texts
    """
    translator = TextTranslator(model_name=model_name, device=device)
    translator.load_model()

    if isinstance(texts, str):
        result = translator.translate(texts, src_lang, tgt_lang)
    else:
        result = translator.translate_batch(texts, src_lang, tgt_lang)

    translator.unload_model()
    return result


# Example usage
if __name__ == "__main__":
    # Test translation
    test_texts = [
        "Hello, how are you?",
        "This is a test of the translation system.",
        "The weather is nice today.",
        "I love learning new languages."
    ]

    print("Text Translation Test\n")
    print("=" * 70)

    translator = TextTranslator(model_name="facebook/nllb-200-3.3B")
    translator.load_model()

    # Translate from English to German
    print("\nTranslating English → German:")
    translations = translator.translate_batch(
        test_texts,
        src_lang="eng_Latn",
        tgt_lang="deu_Latn",
        batch_size=2
    )

    print("\nResults:")
    for original, translated in zip(test_texts, translations):
        print(f"  EN: {original}")
        print(f"  DE: {translated}\n")

    translator.unload_model()
