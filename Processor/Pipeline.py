from Processor.CommonCore.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.Grapheme2PhonemeConverter import \
                             Grapheme2PhonemeConverter as G2P
from Processor.DeepGraphemizer.Phoneme2GraphemeConverter import \
                             Phoneme2GraphemeConverter as P2G
from Processor.PhonemeMapper.Mapper import PhonemeMap
from Processor.Reconstructor.Reconstructor import Reconstructor
from Processor.Tokenizer.Tokenizer import Tokenizer

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
        # for t, p, rp, g in zip(token_text, phonemes, ro_phonemes, graphemes):
        #     t_str = truncate(t, COL_WIDTH)
        #     p_str = truncate(p, COL_WIDTH)
        #     rp_str = truncate(rp, COL_WIDTH)
        #     g_str = truncate(g, COL_WIDTH)
        #     print(
        #         f"{t_str.ljust(COL_WIDTH)}--[G2P]-> "
        #         f"{p_str.ljust(COL_WIDTH)}--[Map]-> "
        #         f"{rp_str.ljust(COL_WIDTH)}--[P2G]-> "
        #         f"{g_str.ljust(COL_WIDTH)}"
        #     )
        return Reconstructor.apply(tokens, graphemes)
