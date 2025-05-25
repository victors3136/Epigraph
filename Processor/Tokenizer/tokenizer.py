import spacy
from spacy.tokens import Doc
from Processor.Domain.supported_language import SupportedLanguage


class Tokenizer:
    _italian = spacy.load("it_core_news_lg")
    _spanish = spacy.load("es_dep_news_trf")

    @staticmethod
    def apply(lang: SupportedLanguage, text: str) -> Doc:
        match lang:
            case SupportedLanguage.Italian:
                return Tokenizer._italian(text)
            case SupportedLanguage.Spanish:
                return Tokenizer._spanish(text)
            case _:
                assert False