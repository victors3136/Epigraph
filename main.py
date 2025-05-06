from expose_deep_phonemizer_module import expose_dp
from Processor.Domain.supported_language import SupportedLanguage
from Processor.DeepGraphemizer.phoneme2grapheme_converter import Phoneme2GraphemeConverter as P2G
from Processor.DeepPhonemizer.grapheme2phoneme_converter import Grapheme2PhonemeConverter as G2P
from Processor.PhonemeMapper.mapper import PhonemeMap
from Processor.pipeline import Pipeline
from Loader.cv_loader import Loader
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

    # p_ita = Pipeline(SupportedLanguage.Italian)
    # text = """
    # Nel mezzo del cammin di nostra vita
    # mi ritrovai per una selva oscura,
    # ché la diritta via era smarrita.
    # Ahi quanto a dir qual era è cosa dura
    # esta selva selvaggia e aspra e forte
    # che nel pensier rinova la paura!
    # Tant' è amara che poco è più morte;
    # ma per trattar del ben ch'i' vi trovai,
    # dirò de l'altre cose ch'i' v'ho scorte
    # """
    # result = p_ita(text)
    # print("[===ITA===]")
    # print("\nBase:")
    # print(text)
    # print("\nResult:")
    # print(result)
    # print("[=========]\n")
    # p_spa = Pipeline(SupportedLanguage.Spanish)
    # text = """
    # Desocupado Lector: sin juramento me podrás creer que quisiera que este
    # Libro, como hijo del entendimiento, fuera el más hermoso, el más gallardo y
    # más discreto que pudiera imaginarse; pero no he podido yo contravenir al
    # orden de Naturaleza: que en ella cada cosa engendra su semejante. Y así, ¿qué podrá en-
    # gendrar el estéril y mal cultivado ingenio mío
    # """ 
    # result = p_spa(text)
    # print("[===SPA===]")
    # print("\nBase:")
    # print(text)
    # print("\nResult:")
    # print(result)
    # print("[=========]\n")

    loader = Loader(it_fraction=0.1, es_fraction=0.1)
    dataset = loader.load(20)

    print(f"Train size: {len(dataset['train'])}")
    for training_example in dataset["train"]:
        print(f"Sample: {training_example['sentence']}")

    warnings.filterwarnings("ignore",
                            message="Implicitly cleaning up <TemporaryDirectory*",
                            category=ResourceWarning)