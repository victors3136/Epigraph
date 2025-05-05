from expose_deep_phonemizer_module import expose_dp
from Processor.CommonCore.supported_language import SupportedLanguage
from Processor.DeepGraphemizer.Phoneme2GraphemeConverter import Phoneme2GraphemeConverter as P2G
from Processor.DeepPhonemizer.Grapheme2PhonemeConverter import Grapheme2PhonemeConverter as G2P
from Processor.PhonemeMapper.Mapper import PhonemeMap
from Processor.Pipeline import Pipeline
import warnings

expose_dp()

ita_phonemizer = G2P(
    SupportedLanguage.Italian,
    "Processor/DeepPhonemizer/g2p_latin_models/"
)

spa_phonemizer = G2P(
    SupportedLanguage.Spanish,
    "Processor/DeepPhonemizer/g2p_latin_models/"
)
graphemizer = P2G("Processor/DeepGraphemizer/p2g_romanian_model")

def process_ita(base: list[str]):
    ita = ita_phonemizer(base)
    ita_p = PhonemeMap.apply(SupportedLanguage.Italian, ita)
    ita_r = graphemizer(ita_p)
    for word, i1, i2, i3 in zip(base, ita, ita_p, ita_r):
        print(f"{word} --[Italian G2P]-> {i1} --[It2Ro Map]-> {i2} --[Romanian P2G]-> {i3}")

def process_spa(base: list[str]):
    spa = spa_phonemizer(base)
    spa_p = PhonemeMap.apply(SupportedLanguage.Spanish, spa)
    spa_r = graphemizer(spa_p)
    for word, i1, i2, i3 in zip(base, spa, spa_p, spa_r):
        print(f"{word} --[Spanish G2P]-> {i1} --[Sp2Ro Map]-> {i2} --[Romanian P2G]-> {i3}")

if __name__ == "__main__":

    # process_ita(["capra", "calca", "piatra"])
    # process_spa(["capra", "calca", "piatra"])

    # process_ita(["dies", "irae", "dies", "illa"])
    # process_spa(["dies", "irae", "dies", "illa"])
    
    # process_ita(["E", "se", "muoio", "da", "partigiano", "Tu", "mi", "devi", "seppellir"])
    # process_ita(["Gli", "angeli", "dell’inferno", "cantano", "piano"])
        
    # process_ita(["La", "pentora", "spinava", "il", "dosciglio", "del", "cravone"])

    # process_spa(["Para", "bailar", "la", "bamba", "se", "necesita", "Una", "poca", "de", "gracia"])
    # process_spa(["Aquella", "estrella", "brillaba", "en", "silencio"])
    # process_spa(["Unas", "vetilas", "de", "monchero", "cruzaron", "el", "bragón"])

    p_ita = Pipeline(SupportedLanguage.Italian)
    text = "Zambo Gimmi ha detto:\n" \
        "Una mattina mi son svegliato\n" \
        "O bella ciao, bella ciao, bella ciao ciao ciao\n" \
        "Una mattina mi son svegliato\n" \
        "E ho trovato l'invasor\n" \
        "O partigiano porta mi via\n" \
        "O bella ciao, bella ciao, bella ciao ciao ciao\n" \
        "O partigiano porta mi via\n" \
        "Che mi sento di morir"
    result = p_ita(text)
    print("[===ITA===]")
    print("\nBase:")
    print(text)
    print("\nResult:")
    print(result)
    print("[=========]\n")
    p_spa = Pipeline(SupportedLanguage.Spanish)
    text = "Zambo Gimmi ha dichio:\n" \
            "La cucaracha, la cucaracha\n" \
            "Ya no puede caminar\n" \
            "Porque no tiene, porque le falta\n" \
            "Una pata para andar\n" \
            "Una cucaracha grande\n" \
            "Se pasea en la cocina\n" \
            "Y la chancla de mi madre\n" \
            "Le ha quitado una patita " 
    result = p_spa(text)
    print("[===SPA===]")
    print("\nBase:")
    print(text)
    print("\nResult:")
    print(result)
    print("[=========]\n")

    warnings.filterwarnings("ignore",
                            message="Implicitly cleaning up <TemporaryDirectory*",
                            category=ResourceWarning)
