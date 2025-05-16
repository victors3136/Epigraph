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
from Loader.cv_loader import Loader


expose_dp()

if __name__ == "__main__":
    try:
        loader = Loader(it_fraction=0.00, es_fraction=0.00)
        for i in range(4_000, 5_001, 500):
            dataset = loader.load(i)
        dataset.push_to_hub("victors3136/dataset-5k-00it-00sp")
    except IndexError as ie:
        print(f"\nToo much data requested :(\n{ie}")
    # loader = Loader(it_fraction=0.05, es_fraction=0.05)
    # loader.load(5_000).push_to_hub("victors3136/dataset-5k-05it-05sp")
    # loader = Loader(it_fraction=0.15, es_fraction=0.15)
    # loader.load(5_000).push_to_hub("victors3136/dataset-5k-15it-15sp")
    # loader = Loader(it_fraction=0.25, es_fraction=0.25)
    # loader.load(5_000).push_to_hub("victors3136/dataset-5k-25it-25sp")
    # loader = Loader(it_fraction=0.35, es_fraction=0.35)
    # loader.load(5_000).push_to_hub("victors3136/dataset-5k-35it-35sp")
    # loader = Loader(it_fraction=0.15, es_fraction=0.35)
    # loader.load(5_000).push_to_hub("victors3136/dataset-5k-15it-35sp")
    # loader = Loader(it_fraction=0.35, es_fraction=0.15)
    # loader.load(5_000).push_to_hub("victors3136/dataset-5k-35it-15sp")
