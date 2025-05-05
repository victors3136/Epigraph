import re
from spacy.tokens import Doc

class Reconstructor:
    def normalize_apostrophes(text: str) -> str:
        # Replace apostrophes between two word characters (elision) with hyphen
        # In Italian and Spanish we have constructs like:
        # l'avevo 
        # And we would like to turn it to a more Romanian-like:
        # l-avevo
        text = re.sub(r"(?<=\w)'(?=\w)", "-", text)
    
        # Step 2: Skip whitepsace between 2 words when there is an apostrophe
        text = re.sub(r"'\s", "'", text)

        return text

    @classmethod
    def apply(cls, baseDoc: Doc, graphemes: list[str]) -> str:
        out_words = []
        grapheme_iter = iter(graphemes)
        for token in baseDoc:
            if re.match(r"^\w+$", token.text):
                new_word = next(grapheme_iter)

                if token.text.istitle():
                    new_word = new_word.capitalize()
                elif token.text.isupper():
                    new_word = new_word.upper()
                out_words.append(new_word + token.whitespace_)
            else:
                out_words.append(token.text + token.whitespace_)

        return cls.normalize_apostrophes("".join(out_words))