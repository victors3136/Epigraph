from expose_deep_phonemizer_module import expose_dp
from Processor.Domain.supported_language import SupportedLanguage
from Processor.pipeline import Pipeline

expose_dp()

if __name__ == "__main__":
    p_ita = Pipeline(SupportedLanguage.Italian)
    text = """
    Nel mezzo del cammin di nostra vita
    mi ritrovai per una selva oscura,
    ché la diritta via era smarrita.
    Ahi quanto a dir qual era è cosa dura
    esta selva selvaggia e aspra e forte
    che nel pensier rinova la paura!
    Tant' è amara che poco è più morte;
    ma per trattar del ben ch'i' vi trovai,
    dirò de l'altre cose ch'i' v'ho scorte
    """
    result = p_ita(text)
    print("[===ITA===]")
    print("\nBase:")
    print(text)
    print("\nResult:")
    print(result)
    print("[=========]\n")
    p_spa = Pipeline(SupportedLanguage.Spanish)
    text = """
    Desocupado Lector: sin juramento me podrás creer que quisiera que este
    Libro, como hijo del entendimiento, fuera el más hermoso, el más gallardo y
    más discreto que pudiera imaginarse; pero no he podido yo contravenir al
    orden de Naturaleza: que en ella cada cosa engendra su semejante. Y así, ¿qué podrá en-
    gendrar el estéril y mal cultivado ingenio mío
    """ 
    result = p_spa(text)
    print("[===SPA===]")
    print("\nBase:")
    print(text)
    print("\nResult:")
    print(result)
    print("[=========]\n")
