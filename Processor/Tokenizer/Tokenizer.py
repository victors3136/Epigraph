import spacy
from spacy.tokens import Doc
from Processor.Domain.supported_language import SupportedLanguage


class Tokenizer:
    _Italian = spacy.load("it_core_news_lg")
    _Spanish = spacy.load("es_dep_news_trf")

    @classmethod
    def apply(cls, lang: SupportedLanguage, text: str) -> Doc:
        match(lang):
            case SupportedLanguage.Italian:
                return cls._Italian(text)
            case SupportedLanguage.Spanish:
                return cls._Spanish(text)