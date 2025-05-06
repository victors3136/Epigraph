import torch
import warnings

from Processor.Domain.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.dp.phonemizer import Phonemizer


class Grapheme2PhonemeConverter:
    def __init__(self,
                 language: SupportedLanguage,
                 path_prefix: str = "./g2p_latin_models/"):
        print(f"Loading G2P for {language} ... ")
        self.language = language
        device = "cuda" if torch.cuda.is_available() else "cpu"
        warnings.filterwarnings("ignore",
                                message="enable_nested_tensor is True, but self.use_nested_tensor is False*",
                                category=UserWarning)
        self.model = Phonemizer.from_checkpoint(path_prefix + language.to_best_model(),
                                                device=device)
        warnings.resetwarnings()
        print(f"Loaded G2P for {language}! ")

    def __call__(self, words: list[str]) -> list[str]:
        result = self.model(words, lang=str(self.language))
        return result if isinstance(result, list) else [result]
