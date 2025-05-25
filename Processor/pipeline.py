from Processor.Domain.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.grapheme2phoneme_converter import \
                             Grapheme2PhonemeConverter as g2pConverter
from Processor.DeepGraphemizer.phoneme2grapheme_converter import \
                             Phoneme2GraphemeConverter as p2gConverter
from Processor.PhonemeMapper.mapper import PhonemeMap
from Processor.Reconstructor.reconstructor import Reconstructor
from Processor.Tokenizer.tokenizer import Tokenizer

import re

p2g = p2gConverter("./Processor/DeepGraphemizer/p2g_romanian_model")
lang_2_g2p_map = {}

class Pipeline:
    def __init__(self, lang: SupportedLanguage):
        self.lang = lang
        if lang not in lang_2_g2p_map.keys():
            lang_2_g2p_map[lang] = g2pConverter(lang, "./Processor/DeepPhonemizer/g2p_latin_models/")
        self.g2p = lang_2_g2p_map[lang]
        self.p2g = p2g

    def __call__(self, text: str) -> str:
        tokens = Tokenizer.apply(self.lang, text)
        word_tokens = [token.text for token in tokens if re.match(r"^\w+$", token.text)]
        phonemes = self.g2p(word_tokens)
        ro_phonemes = PhonemeMap.apply(self.lang, phonemes)
        graphemes = self.p2g(ro_phonemes)
        return Reconstructor.apply(tokens, graphemes)
