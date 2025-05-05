import re
from spacy.tokens import Doc

class Reconstructor:
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

        return "".join(out_words)