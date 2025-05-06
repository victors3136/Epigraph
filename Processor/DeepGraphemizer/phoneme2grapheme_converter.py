import torch
from transformers.models.encoder_decoder import EncoderDecoderModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging as hf_logging


class Phoneme2GraphemeConverter:
    def __init__(self,
                 model_dir: str = "./p2g_romanian_model"):
        print("Initializing P2G ... ")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        initial_log_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        print("Loading P2G ... ")
        self.model = EncoderDecoderModel \
            .from_pretrained(f"{model_dir}/final_model")
        self.phoneme_tokenizer = PreTrainedTokenizerFast \
            .from_pretrained(f"{model_dir}/phoneme_tokenizer")
        self.grapheme_tokenizer = PreTrainedTokenizerFast \
            .from_pretrained(f"{model_dir}/grapheme_tokenizer")
        hf_logging.set_verbosity(initial_log_verbosity)
        self.model.to(self.device)

        self.model.eval()
        print("P2G initialized!")

    @classmethod
    def __deformat(cls, phoneme: str):
        return "".join(phoneme.split())

    def __call__(self, phonemes: str | list[str]) -> list[str] | str:
        if isinstance(phonemes, list):
            return [self.__call__(p) for p in phonemes]

        inputs = self.phoneme_tokenizer(phonemes, return_tensors="pt") \
            .to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=50)
        word = self.grapheme_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        return self.__deformat(word)
