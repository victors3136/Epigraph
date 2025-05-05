from Processor.CommonCore.supported_language import SupportedLanguage
from Processor.PhonemeMapper.ItalianPhonemeMapper import ItalianPhonemeMapper
from Processor.PhonemeMapper.SpanishPhonemeMapper import SpanishPhonemeMapper


def __get_mapper(language: SupportedLanguage):
    match language:
        case SupportedLanguage.Italian:
            return ItalianPhonemeMapper
        case SupportedLanguage.Spanish:
            return SpanishPhonemeMapper
        case _:
            raise Exception(f"Unknown language: {language}")


def phoneme_map(language: SupportedLanguage, phoneme: str) -> str:
    i = 0
    mapper = __get_mapper(language)
    result = []
    while i < len(phoneme):
        if phoneme[i] == " ":
            i += 1
            continue
        matched = False
        for key in mapper.keys():
            if phoneme[i:i + len(key)] == key:
                result.append(mapper.dict()[key])
                i += len(key)
                matched = True
                break
        if not matched:
            raise ValueError(f"Unrecognized phoneme sequence at position {i}: '{phoneme[i:]}'")
    return ''.join(result)
