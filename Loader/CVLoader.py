import random
from datasets import load_dataset
from Processor.Pipeline import Pipeline
from Processor.Domain.supported_language import SupportedLanguage

DATASET = "mozilla-foundation/common_voice_11_0"

class Loader:
    def __init__(self, it_fraction: float, es_fraction: float, seed: int = 42):
        assert 0 <= it_fraction <= 1
        assert 0 <= es_fraction <= 1
        self.it_fraction = it_fraction
        self.es_fraction = es_fraction
        self.pipeline_it = Pipeline(SupportedLanguage.Italian)
        self.pipeline_es = Pipeline(SupportedLanguage.Spanish)
        self.random = random.Random(seed)

    def load(self, n_samples: int = 10_000):
        print("Loading Common Voice Romanian...")
        ro_ds = load_dataset(DATASET, "ro", split="train")
        print("Loading Common Voice Italian...")
        it_ds = load_dataset(DATASET, "it", split="train")
        print("Loading Common Voice Spanish...")
        es_ds = load_dataset(DATASET, "es", split="train")

        ro_samples = [x for x in ro_ds if x["audio"] and x["sentence"]]
        it_samples = [x for x in it_ds if x["audio"] and x["sentence"]]
        es_samples = [x for x in es_ds if x["audio"] and x["sentence"]]

        self.random.shuffle(ro_samples)

        n_ro = n_samples
        n_it = int(self.it_fraction * n_ro)
        n_es = int(self.es_fraction * n_ro)

        val_size = int(0.1 * n_ro)
        test_size = int(0.1 * n_ro)

        val_set = ro_samples[:val_size]
        test_set = ro_samples[val_size:val_size + test_size]
        ro_train = ro_samples[val_size + test_size:val_size + test_size + (n_ro - val_size - test_size)]

        it_data = self.random.sample(it_samples, n_it)
        es_data = self.random.sample(es_samples, n_es)

        for sample in it_data:
            sample["sentence"] = self.pipeline_it(sample["sentence"])
        for sample in es_data:
            sample["sentence"] = self.pipeline_es(sample["sentence"])

        train_set = ro_train + it_data + es_data
        self.random.shuffle(train_set)

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }
