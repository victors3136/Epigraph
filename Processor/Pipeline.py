from Processor.Domain.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.grapheme2phoneme_converter import \
                             Grapheme2PhonemeConverter as G2P
from Processor.DeepGraphemizer.phoneme2grapheme_converter import \
                             Phoneme2GraphemeConverter as P2G
from Processor.PhonemeMapper.mapper import PhonemeMap
from Processor.Reconstructor.reconstructor import Reconstructor
from Processor.Tokenizer.tokenizer import Tokenizer

import re

def truncate(text, max_len):
    return text if len(text) <= max_len else text[:max_len - 1] + "…"

COL_WIDTH = 18

class Pipeline:
    def __init__(self, lang: SupportedLanguage):
        self.lang = lang
        self.g2p = G2P(lang, "./Processor/DeepPhonemizer/g2p_latin_models/")
        self.p2g = P2G("./Processor/DeepGraphemizer/p2g_romanian_model")

    def __call__(self, text: str) -> str:
        tokens = Tokenizer.apply(self.lang, text)
        
        word_tokens = [token for token in tokens if re.match(r"^\w+$", token.text)]

        token_text = [token.text for token in word_tokens]
        
        phonemes = self.g2p(token_text)
        ro_phonemes = PhonemeMap.apply(self.lang, phonemes)
        graphemes = self.p2g(ro_phonemes)
        return Reconstructor.apply(tokens, graphemes)
