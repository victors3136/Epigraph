from Processor.CommonCore.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.Grapheme2PhonemeConverter import \
                             Grapheme2PhonemeConverter as G2P
from Processor.DeepGraphemizer.Phoneme2GraphemeConverter import \
                             Phoneme2GraphemeConverter as P2G
from Processor.PhonemeMapper.Mapper import phoneme_map

class Pipeline:
    def __init__(self, lang: SupportedLanguage):
        self.lang = lang
        g2p = G2P(lang, "./DeepPhonemizer/g2p_latin_models/")
        p2g = P2G("./DeepGraphemizer/p2g_romanian_model")

    def __call__(self, text: str) -> str:
        phonemes = g2p(text)
        ro_phonemes  == phoneme_map(self.lang, phonemes)
        graphemes = p2g(ro_phonemes)
        print(f"{text} --[G2P]-> {phonemes} --[Map]-> {ro_phonemes} --[P2G]-> {graphemes}")
        return graphemes