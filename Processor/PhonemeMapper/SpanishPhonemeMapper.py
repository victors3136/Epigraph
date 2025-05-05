class SpanishPhonemeMapper:
    __Sp2Ro = {
        # vowels
        "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
        # consonants
        "b": "b", "d": "d", "t": "t", "k": "k", "g": "ɡ", "p": "p", "f": "f", "ɡ" : "ɡ",
        "s": "s", "z": "z", "m": "m", "n": "n", "ɲ": "nj", "ʎ": "lj", "l": "l", "j": "j",
        "r": "r", "ɾ": "r", "tʃ": "tʃ", "x": "h", "θ": "s", "ʃ": "ʃ", "ʝ": "j"
    }
    # sorted so that we check for longer constructs first
    # e.g. check for 'tʃ' before checking for 't'
    __Sp2RoKeys = sorted(__Sp2Ro.keys(), key=len, reverse=True)

    @classmethod
    def keys(cls) -> list:
        return cls.__Sp2RoKeys

    @classmethod
    def dict(cls) -> dict:
        return cls.__Sp2Ro
