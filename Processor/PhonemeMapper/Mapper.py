from Processor.Domain.supported_language import SupportedLanguage
from Processor.PhonemeMapper.ItalianPhonemeMapper import ItalianPhonemeMapper
from Processor.PhonemeMapper.SpanishPhonemeMapper import SpanishPhonemeMapper



class PhonemeMap:
    @classmethod
    def __get_mapper(cls, language: SupportedLanguage):
        match language:
            case SupportedLanguage.Italian:
                return ItalianPhonemeMapper
            case SupportedLanguage.Spanish:
                return SpanishPhonemeMapper

    @classmethod
    def apply(cls, language: SupportedLanguage, phoneme: str | list[str]) -> str | list[str]:
        if isinstance(phoneme, list):
            return [cls.apply(language, p) for p in phoneme]
        i = 0
        mapper = cls.__get_mapper(language)
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
                print(f"Unrecognized phoneme sequence at position {i}: '{phoneme[i:]}'")
                i += 1
        return ''.join(result)