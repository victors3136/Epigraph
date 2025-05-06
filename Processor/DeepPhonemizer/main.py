from Processor.Domain.supported_language import SupportedLanguage
from Processor.DeepPhonemizer.grapheme2phoneme_converter import Grapheme2PhonemeConverter

if __name__ == "__main__":
    it = Grapheme2PhonemeConverter(SupportedLanguage.Italian)

    print(it("ciao"))

    print(it(["georgiana", "arrivederci"]))

    print(it(["Gli", "angeli", "dell’inferno", "cantano", "piano"]))

    print(it(["me", "chiama", "giorgio", "chevalero"]))

    print(it(["mi", "piace", "quando", "volves"]))

    es = Grapheme2PhonemeConverter(SupportedLanguage.Spanish)

    print(es("churros"))

    print(es("guarroces"))

    print(es(["me", "llamo", "charli"]))

    print(es(["mi", "gusta", "cuando", "vuelves"]))

