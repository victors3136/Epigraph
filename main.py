import warnings
warnings.filterwarnings("ignore",
                        message="Passing a tuple of `past_key_values` is deprecated and will be removed*",
                        category=UserWarning)
warnings.filterwarnings("ignore",
                        message="Moving the following attributes in the config to the generation config*",
                        category=UserWarning)
warnings.filterwarnings("ignore",
                        message="Implicitly cleaning up <TemporaryDirectory*",
                        category=ResourceWarning)

from expose_deep_phonemizer_module import expose_dp
from Orchestrator.fine_tuner import WhisperFinetuner


expose_dp()

if __name__ == "__main__":
    tuner = WhisperFinetuner(0.1, 0.1, "./models/010142/")
    tuner.train(10, 3, 2)