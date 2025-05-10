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

    def __call__(self, phonemes: str | list[str]) -> list[str] | str | None:
        if isinstance(phonemes, str):
            phonemes = [phonemes]
            single_input = True
        else:
            single_input = False

        inputs = self.phoneme_tokenizer(phonemes, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs["input_ids"] = inputs["input_ids"].long()

        with torch.no_grad():
            try:
                outputs = self.model.generate(**inputs, max_length=50)
            except RuntimeError as re:
                if "Expected tensor for argument" in str(re):
                    return None if single_input else [None] * len(phonemes)
                else:
                    raise re

        decoded = self.grapheme_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        reformatted = [self.__deformat(word) for word in decoded]

        return reformatted[0] if single_input else reformatted

