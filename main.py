from Processor.CommonCore import SupportedLanguage
from Processor.DeepGraphemizer import Phoneme2GraphemeConverter
from Processor.DeepPhonemizer.Grapheme2PhonemeConverter import Grapheme2PhonemeConverter
from Processor.PhonemeMapper import phoneme_map

if __name__ == "__main__":
    ita_phonemizer = Grapheme2PhonemeConverter(
        SupportedLanguage.Italian,
        "Processor/DeepPhonemizer/g2p_latin_models/"
    )
    spa_phonemizer = Grapheme2PhonemeConverter(
        SupportedLanguage.Spanish,
        "Processor/DeepPhonemizer/g2p_latin_models/"
    )
    graphemizer = Phoneme2GraphemeConverter("Processor/DeepGraphemizer/p2g_romanian_model")

    for word in ["capra", "calca", "piatra",        # Romanian (no diacritics)
                 "dies", "irae", "dies", "illa"]:   # Latin
        ita = ita_phonemizer(word)
        ita_p = phoneme_map(SupportedLanguage.Italian, ita)
        ita_r = graphemizer(ita_p)
        print(f"{word} --[Italian G2P]-> {ita} --[It2Ro Map]-> {ita_p} --[Romanian P2G]-> {ita_r}")

        spa = spa_phonemizer(word)
        spa_p = phoneme_map(SupportedLanguage.Spanish, spa)
        spa_r = graphemizer(spa_p)
        print(f"{word} --[Spanish G2P]-> {spa} --[Sp2Ro Map]-> {spa_p} --[Romanian P2G]-> {spa_r}")

    for word in ["E", "se", "muoio", "da", "partigiano", "Tu", "mi", "devi", "seppellir"]:
        ita = ita_phonemizer(word)
        ita_p = phoneme_map(SupportedLanguage.Italian, ita)
        ita_r = graphemizer(ita_p)
        print(f"{word} --[Italian G2P]-> {ita} --[It2Ro Map]-> {ita_p} --[Romanian P2G]-> {ita_r}")

    for word in ["Gli", "angeli", "dell’inferno", "cantano", "piano"]:
        ita = ita_phonemizer(word)
        ita_p = phoneme_map(SupportedLanguage.Italian, ita)
        ita_r = graphemizer(ita_p)
        print(f"{word} --[Italian G2P]-> {ita} --[It2Ro Map]-> {ita_p} --[Romanian P2G]-> {ita_r}")

    for word in ["La", "pentora", "spinava", "il", "dosciglio", "del", "cravone"]:
        ita = ita_phonemizer(word)
        ita_p = phoneme_map(SupportedLanguage.Italian, ita)
        ita_r = graphemizer(ita_p)
        print(f"{word} --[Italian G2P]-> {ita} --[It2Ro Map]-> {ita_p} --[Romanian P2G]-> {ita_r}")


    for word in ["Para", "bailar", "la", "bamba", "se", "necesita", "Una", "poca", "de", "gracia"]:
        spa = spa_phonemizer(word)
        spa_p = phoneme_map(SupportedLanguage.Spanish, spa)
        spa_r = graphemizer(spa_p)
        print(f"{word} --[Spanish G2P]-> {spa} --[Sp2Ro Map]-> {spa_p} --[Romanian P2G]-> {spa_r}")

    for word in ["Aquella", "estrella", "brillaba", "en", "silencio"]:
        spa = spa_phonemizer(word)
        spa_p = phoneme_map(SupportedLanguage.Spanish, spa)
        spa_r = graphemizer(spa_p)
        print(f"{word} --[Spanish G2P]-> {spa} --[Sp2Ro Map]-> {spa_p} --[Romanian P2G]-> {spa_r}")

    for word in ["Unas", "vetilas", "de", "monchero", "cruzaron", "el", "bragón"]:
        spa = spa_phonemizer(word)
        spa_p = phoneme_map(SupportedLanguage.Spanish, spa)
        spa_r = graphemizer(spa_p)
        print(f"{word} --[Spanish G2P]-> {spa} --[Sp2Ro Map]-> {spa_p} --[Romanian P2G]-> {spa_r}")
