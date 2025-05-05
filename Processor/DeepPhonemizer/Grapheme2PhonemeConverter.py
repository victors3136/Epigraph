import torch
import warnings

from Processor.CommonCore.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.dp.phonemizer import Phonemizer


class Grapheme2PhonemeConverter:
    def __init__(self,
                 language: SupportedLanguage,
                 path_prefix: str = "./g2p_latin_models/"):
        self.language = language
        device = "cuda" if torch.cuda.is_available() else "cpu"
        warnings.filterwarnings("ignore",
                                message="enable_nested_tensor is True, but self.use_nested_tensor is False*",
                                category=UserWarning)
        self.model = Phonemizer.from_checkpoint(path_prefix + language.to_best_model(),
                                                device=device)
        warnings.resetwarnings()

    def __call__(self, word: str | list[str]) -> str:
        result = self.model(word, lang=str(self.language))
        if isinstance(result, list):
            return " ".join(result)
        return result
