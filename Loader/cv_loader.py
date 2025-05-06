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
    
    @classmethod
    def collect_valid_samples(cls, dataset_iter, max_samples, random_gen):
        collected = []
        for entry in dataset_iter:
            if entry.get("audio") and entry.get("sentence"):
                collected.append(entry)
                if len(collected) >= max_samples:
                    break
        random_gen.shuffle(collected)
        return collected

    def load(self, n_samples: int = 10_000):
        print("Streaming Romanian data...")
        ro_data = self.collect_valid_samples(
            load_dataset(DATASET, "ro", split="train", streaming=True),
            n_samples,
            self.random
        )

        val_size = int(0.1 * n_samples)
        test_size = int(0.1 * n_samples)
        train_size = n_samples - val_size - test_size

        val_set = ro_data[:val_size]
        test_set = ro_data[val_size:val_size + test_size]
        ro_train = ro_data[val_size + test_size:]

        n_it = int(self.it_fraction * train_size)
        n_es = int(self.es_fraction * train_size)

        print("Streaming Italian data...")
        it_data = self.collect_valid_samples(
            load_dataset(DATASET, "it", split="train", streaming=True),
            n_it,
            self.random
        )
        print("Streaming Spanish data...")
        es_data = self.collect_valid_samples(
            load_dataset(DATASET, "es", split="train", streaming=True),
            n_es,
            self.random
        )

        print("Phonetically converting IT...")
        for sample in it_data:
            result = self.pipeline_it(sample["sentence"])
            print(f"{sample['sentence']} --> {result}")
            sample["sentence"] = result
        
        print("Phonetically converting ES...")
        for sample in es_data:
            result = self.pipeline_es(sample["sentence"])
            print(f"{sample['sentence']} --> {result}")
            sample["sentence"] = result

        print("Concattenating data sets...")
        train_set = ro_train + it_data + es_data
        self.random.shuffle(train_set)

        print(f"Train: {len(train_set)}; Validation: {len(val_set)}; Test: {len(test_set)}")

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set
        }