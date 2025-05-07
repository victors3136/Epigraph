import random
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Features, Audio, Value
from Processor.Pipeline import Pipeline
from Processor.Domain.supported_language import SupportedLanguage

DATASET = "mozilla-foundation/common_voice_11_0"
CACHE_DIR = "./cached_dataset"

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
        for entry in tqdm(dataset_iter, desc="Filtering for valid data entries"):
            if entry.get("audio") and entry.get("sentence"):
                collected.append(entry)
                if len(collected) >= max_samples:
                    break
        random_gen.shuffle(collected)
        return collected
    @classmethod
    def simplify(cls, sample):
        return {
            "audio": sample["audio"],
            "sentence": sample["sentence"]
        }

    def load(self, n_samples: int = 10_000):
        if os.path.exists(CACHE_DIR):
            print("Loading cached dataset...")
            data = DatasetDict.load_from_disk(CACHE_DIR)
            total_cached = sum(len(data[split]) for split in data)

            if total_cached >= n_samples:
                print(f"Loaded {total_cached} cached samples.")
                return DatasetDict({
                    split: data[split].select(range(min(n_samples, len(data[split]))))
                    for split in data
                })

            print(f"Cached data insufficient: {total_cached} found, {n_samples} needed.")
            remaining_samples = n_samples - total_cached
        else:
            data = None
            remaining_samples = n_samples

        print("Streaming Romanian data...")
        ro_data = self.collect_valid_samples(
            load_dataset(DATASET, "ro", split="train", streaming=True),
            remaining_samples,
            self.random
        )

        val_size = int(0.1 * remaining_samples)
        test_size = int(0.1 * remaining_samples)
        train_size = remaining_samples - val_size - test_size

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
        for sample in tqdm(it_data, desc="Converting Italian"):
            result = self.pipeline_it(sample["sentence"])
            sample["sentence"] = result

        print("Phonetically converting ES...")
        for sample in tqdm(es_data, desc="Converting Spanish"):
            result = self.pipeline_es(sample["sentence"])
            sample["sentence"] = result

        print("Simplifying data sets...")
        train_set = list(map(self.simplify, ro_train + it_data + es_data))
        val_set = list(map(self.simplify, val_set))
        test_set = list(map(self.simplify, test_set))

        print("Concatenating data sets...")
        self.random.shuffle(train_set)

        features = Features({"audio": Audio(), "sentence": Value("string")})

        print("Building Dataset...")
        new_dataset = DatasetDict({
            "train": Dataset.from_list(train_set, features=features),
            "val": Dataset.from_list(val_set, features=features),
            "test": Dataset.from_list(test_set, features=features),
        })

        if data:
            print("Merging with cached data...")
            for split in ["train", "val", "test"]:
                if split in data:
                    new_dataset[split] = data[split].concatenate(new_dataset[split]) if split in new_dataset else data[split]

        total_size = sum(len(new_dataset[split]) for split in new_dataset)
        assert total_size >= n_samples, f"Total dataset size mismatch: expected at least {n_samples}, got {total_size}"

        print(f"Final sizes — Train: {len(new_dataset['train'])}, Val: {len(new_dataset['val'])}, Test: {len(new_dataset['test'])}")
        print("Saving dataset to disk...")
        new_dataset.save_to_disk(CACHE_DIR)

        return new_dataset
