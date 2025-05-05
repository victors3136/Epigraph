from Processor.CommonCore.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.Grapheme2PhonemeConverter import \
                             Grapheme2PhonemeConverter as G2P
from Processor.DeepGraphemizer.Phoneme2GraphemeConverter import \
                             Phoneme2GraphemeConverter as P2G
from Processor.PhonemeMapper.Mapper import phoneme_map
from Processor.Tokenizer import Tokenizer

class Pipeline:
    def __init__(self, lang: SupportedLanguage):
        self.lang = lang
        self.g2p = G2P(lang, "./Processor/DeepPhonemizer/g2p_latin_models/")
        self.p2g = P2G("./Processor/DeepGraphemizer/p2g_romanian_model")

    def __call__(self, text: str) -> str:
        tokens = Tokenizer.tokenize(self.lang, text)
        token_text = [token.text for token in tokens]
        phonemes = self.g2p(token_text)
        ro_phonemes = phoneme_map(self.lang, phonemes)
        graphemes = self.p2g(ro_phonemes)
        print(f"{text} --[G2P]-> {phonemes} --[Map]-> {ro_phonemes} --[P2G]-> {graphemes}")
        if isinstance(graphemes, list):
            return " ".join(graphemes)
        return graphemes
