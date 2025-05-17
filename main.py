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
from tqdm import tqdm

expose_dp()

if __name__ == "__main__":
    splits = [
                # Low % of foreign data
                (0, 0), (5, 5),
                # Symmetrical % of total data
                (15, 15), (25, 25), (35, 35),
                # Unsymmetrical % of total data, for verifying if any language is more efficient than another
                (5, 25), (25, 5), (15, 35), (35, 15),
                # High % of foreign data
                (50, 0), (0, 50), (50, 50),
            ]
    for it_split, sp_split in tqdm(splits, desc="Generating datasets..."):
        try:
            Loader(it_fraction=it_split/100.0, es_fraction=sp_split/100.0) \
                .load(5_000) \
                .push_to_hub(f"victors3136/dataset-5k-{it_split:02d}it-{sp_split:02d}sp")
        except IndexError as ie:
            print(f"\nToo much data requested :(\n{ie}")
            break