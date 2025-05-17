import random
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Features, Value, Audio
from Processor.pipeline import Pipeline
from Processor.Domain.supported_language import SupportedLanguage
import uuid
import shutil
import json

DatasetUrl = "mozilla-foundation/common_voice_11_0"

SampleInterface = Features({
    "audio": Audio(),
    "sentence": Value("string")
})

class SampleSimplifier:
    def __call__(self, sample: dict) -> dict | None:
        return {"audio": sample["audio"], "sentence": sample["sentence"]} \
            if "audio" in sample and "sentence" in sample \
            else None

sample_simplifier = SampleSimplifier()

def _get_cached_sample_count_by_key(dataset_path: str, key: str) -> int:
    print(f"Reading metadata from {dataset_path}/meta.json for {key}")
    meta_path = os.path.join(dataset_path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
            return data.get(key, 0)
    return 0

def _write_sample_count_for_key(dataset_path: str, count: int, key: str):
    print(f"Writing metadata in {dataset_path}/meta.json for {key}")
    meta_path = os.path.join(dataset_path, "meta.json")
    data = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
    data[key] = count
    with open(meta_path, "w") as f:
        json.dump(data, f)

class IncrementalSampleSelector:
    @classmethod
    def force_write_dataset_to_disk(cls, data: Dataset, dataset_path: str):
        temp_path = dataset_path + "_tmp_" + str(uuid.uuid4())
        print(f"Saving dataset temporarily to {temp_path}")
        data.save_to_disk(temp_path)
        print(f"Removing dataset from {dataset_path}")
        shutil.rmtree(dataset_path)
        print(f"Moving new dataset from {temp_path} to {dataset_path}")
        shutil.move(temp_path, dataset_path)

    @classmethod
    def select(
        cls,
        language_code: str,
        pipeline: Pipeline,
        required_count: int,
        seed: int = 42,
        output_dir: str = "preprocessed_datasets"
    ) -> Dataset :
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = os.path.join(output_dir, language_code)

        if os.path.exists(dataset_path):
            print(f"[{language_code}] Loading existing dataset from disk...")
            existing_dataset = Dataset.load_from_disk(dataset_path, keep_in_memory=False)
            existing_count = _get_cached_sample_count_by_key(output_dir, language_code)
        else:
            existing_dataset = Dataset.from_dict(
                {"audio": [], "sentence": []},
                features=SampleInterface,
            )
            existing_count = 0
        remaining_count = required_count - existing_count
        if remaining_count <= 0:
            print(f"[{language_code}] Already have {existing_count} samples.")
            return existing_dataset.select(range(required_count))

        print(f"[{language_code}] Need {remaining_count} more samples (have {existing_count}).")
        rand = random.Random(seed)

        print(f"[{language_code}] Streaming {remaining_count} new samples...")
    
        stream = load_dataset(DatasetUrl, language_code, split="train", streaming=True)
    
        collected = []
        skipped = 0
        for entry in tqdm(stream, desc=f"Filtering {language_code}"):
            if skipped < existing_count:
                skipped += 1
                continue
            sample = sample_simplifier(entry)
            if sample is not None:
                collected.append(sample)
                if len(collected) >= remaining_count:
                    break

        rand.shuffle(collected)

        print(f"[{language_code}] Phonetically converting {len(collected)} samples...")
        for sample in tqdm(collected, desc=f"Converting {language_code}"):
            sample["sentence"] = pipeline(sample["sentence"])

        new_dataset = Dataset.from_list(collected, features=SampleInterface)
        combined_dataset = concatenate_datasets([existing_dataset, new_dataset])

        print(f"[{language_code}] Saving updated dataset with {len(combined_dataset)} total samples...")
        try:
            combined_dataset.save_to_disk(dataset_path)
        except PermissionError:
            print(f"[{language_code}] Permission error during save. Using fallback strategy.")
            cls.force_write_dataset_to_disk(combined_dataset, dataset_path)

        _write_sample_count_for_key(output_dir, len(combined_dataset), language_code)

        return combined_dataset.select(range(required_count))


class Loader:
    def __init__(self, it_fraction: float, es_fraction: float, seed: int = 42):
        assert 0 <= it_fraction <= 1
        assert 0 <= es_fraction <= 1
        self.italian_fraction = it_fraction
        self.spanish_fraction = es_fraction
        self.italian_pipeline = Pipeline(SupportedLanguage.Italian)
        self.spanish_pipeline = Pipeline(SupportedLanguage.Spanish)
        self.random = random.Random(seed)

    def _load_romanian_data(self, count: int, output_dir: str = "preprocessed_datasets") -> list:
        dataset_path = os.path.join(output_dir, "ro")
        os.makedirs(output_dir, exist_ok=True)

        existing_count = 0
        collected = []

        if os.path.exists(dataset_path):
            print("[ro] Loading cached Romanian samples from disk...")
            existing_dataset = Dataset.load_from_disk(dataset_path)
            existing_count = _get_cached_sample_count_by_key(output_dir, "ro")
            collected = existing_dataset.to_list()

            if existing_count >= count:
                print(f"[ro] Already have {existing_count} samples.")
                return collected[:count]

            print(f"[ro] Found {existing_count}, need {count}. Fetching {count - existing_count} more...")

        else:
            print("[ro] No cached Romanian dataset found. Starting fresh.")

        print(f"[ro] Streaming {count - existing_count} new samples...")
        stream = load_dataset(DatasetUrl, "ro", split="train", streaming=True)

        skipped = 0
        for entry in tqdm(stream, desc="Collecting Romanian samples"):
            if skipped < existing_count:
                skipped += 1
                continue
            simplified = sample_simplifier(entry)
            if simplified is not None:
                collected.append(simplified)
                if len(collected) >= count:
                    break

        print(f"[ro] Saving Romanian dataset with {len(collected)} samples to disk...")
        Dataset.from_list(collected, features=SampleInterface).save_to_disk(dataset_path)
        _write_sample_count_for_key(output_dir, len(collected), "ro")

        return collected[:count]

    def _split_romanian_data(self, data: list) -> tuple[Dataset, Dataset, Dataset]:
        val_size = int(0.1 * len(data))
        test_size = int(0.1 * len(data))

        validation_set = data[:val_size]
        test_set = data[val_size:val_size + test_size]
        training_set = data[val_size + test_size:]

        return Dataset.from_list(training_set, features=SampleInterface), \
               Dataset.from_list(validation_set, features=SampleInterface), \
               Dataset.from_list(test_set, features=SampleInterface)


    def _load_augmented_data(self, train_size: int, output_dir: str = "preprocessed_datasets") -> Dataset:
        it_count = int(self.italian_fraction * train_size)
        es_count = int(self.spanish_fraction * train_size)

        print("Loading Italian data...")
        italian_data = IncrementalSampleSelector.select(
            "it",
            self.italian_pipeline,
            it_count,
            seed=self.random.randint(0, 10_000),
            output_dir=output_dir
        )

        print("Loading Spanish data...")
        spanish_data = IncrementalSampleSelector.select(
            "es",
            self.spanish_pipeline,
            es_count,
            seed=self.random.randint(0, 10_000),
            output_dir=output_dir
        )

        return concatenate_datasets([italian_data, spanish_data])

    def _build_dataset(self, train, val, test) -> DatasetDict:
        print("Running some sanity checks...")
        assert all(sample is not None for sample in train), "Training set contains None samples!"
        assert all(sample is not None for sample in val), "Validation set contains None samples!"
        assert all(sample is not None for sample in test), "Test set contains None samples!"

        dataset = DatasetDict({
            "train": Dataset.from_list(train, features=SampleInterface),
            "val": Dataset.from_list(val, features=SampleInterface),
            "test": Dataset.from_list(test, features=SampleInterface),
        })

        total_size = sum(len(dataset[split]) for split in dataset)
        assert total_size >= len(train) + len(val) + len(test), "Dataset size mismatch."

        print(f"Final sizes — Train: {len(dataset['train'])}, Val: {len(dataset['val'])}, Test: {len(dataset['test'])}")
        return dataset

    def load(self, romanian_sample_count: int = 10_000, output_dir: str="preprocessed_datasets") -> DatasetDict:
        romanian_data = self._load_romanian_data(romanian_sample_count)
        romanian_training_split, validation_split, testing_split = self._split_romanian_data(romanian_data)
        augmented_data = self._load_augmented_data(len(romanian_training_split), output_dir=output_dir).to_list()
        train = concatenate_datasets([romanian_training_split, augmented_data]).to_list()

        return Dataset.from_list(self._build_dataset(train, validation_split, testing_split))

    @classmethod
    def cleanup(cls, dir_name: str):
        print(f"Cleaning {dir_name} up...")
        shutil.rmtree(dir_name, ignore_errors=True)