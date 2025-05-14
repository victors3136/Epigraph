import random
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Audio, Features, Value
from Processor.pipeline import Pipeline
from Processor.Domain.supported_language import SupportedLanguage
import uuid
import shutil

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

class IncrementalSampleSelector:
    @classmethod
    def force_write_dataset_to_disk(cls, data: Dataset, dataset_path: str):
        temp_path = dataset_path + "_tmp_" + str(uuid.uuid4())
        print(f"Saving dataset temporarily to {temp_path}")
        data.save_to_disk(temp_path)
        print(f"Removing dataset from {dataset_path}")
        shutil.rmtree(dataset_path)
        print(f"Moving new dataset to {temp_path}")
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
            existing_dataset = Dataset.load_from_disk(dataset_path)
        else:
            existing_dataset = Dataset.from_dict(
                {"audio": [], "sentence": []},
                features=SampleInterface
            )
        existing_count = len(existing_dataset)
        remaining_count = required_count - existing_count
        if remaining_count <= 0:
            print(f"[{language_code}] Already have {existing_count} samples.")
            return existing_dataset.select(range(required_count))

        print(f"[{language_code}] Need {remaining_count} more samples (have {existing_count}).")
        rand = random.Random(seed)

        print(f"[{language_code}] Streaming {remaining_count} new samples...")
        new_samples = []
        iterator = load_dataset(DatasetUrl, language_code, split="train", streaming=True)

        for entry in tqdm(iterator, desc=f"Filtering {language_code}"):
            sample = sample_simplifier(entry)
            if sample is not None:
                new_samples.append(sample)
                if len(new_samples) >= remaining_count:
                    break

        rand.shuffle(new_samples)

        print(f"[{language_code}] Phonetically converting...")
        for sample in tqdm(new_samples, desc=f"Converting {language_code}"):
            sample["sentence"] = pipeline(sample["sentence"])

        new_dataset = Dataset.from_list(new_samples, features=existing_dataset.features)

        combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
        print("Saving to disk ...")
        try:
            combined_dataset.save_to_disk(dataset_path)
            print(f"[{language_code}] Updated dataset saved with {len(combined_dataset)} samples.")
        except PermissionError:
            print(f"Failed to save to {dataset_path} due to an older version of this dataset beign already present...")
            cls.force_write_dataset_to_disk(combined_dataset, dataset_path)


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

    def _load_romanian_data(self, count: int):
        print("Streaming Romanian data...")
        iterator = load_dataset(DatasetUrl, "ro", split="train", streaming=True)
        collected = []
        for sample in tqdm(iterator, desc="Collecting Romanian samples"):
            simplified = sample_simplifier(sample)
            if simplified is not None:
                collected.append(simplified)
                if len(collected) >= count:
                    break

        self.random.shuffle(collected)
        return collected

    def _split_romanian_data(self, data: list) -> tuple[Dataset, Dataset, Dataset]:
        val_size = int(0.1 * len(data))
        test_size = int(0.1 * len(data))

        validation_set = data[:val_size]
        test_set = data[val_size:val_size + test_size]
        training_set = data[val_size + test_size:]

        return training_set, validation_set, test_set


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
        ).to_list()

        print("Loading Spanish data...")
        spanish_data = IncrementalSampleSelector.select(
            "es",
            self.spanish_pipeline,
            es_count,
            seed=self.random.randint(0, 10_000),
            output_dir=output_dir
        ).to_list()

        return italian_data + spanish_data

    def _build_dataset(self, train, val, test) -> DatasetDict:
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
        augmented_data = self._load_augmented_data(len(romanian_training_split), output_dir=output_dir)
        train = romanian_training_split + augmented_data

        print("Shuffling training data set...")
        self.random.shuffle(train)

        return self._build_dataset(train, validation_split, testing_split)

    @classmethod
    def cleanup(cls, dir_name: str):
        print(f"Cleaning {dir_name} up...")
        shutil.rmtree(dir_name, ignore_errors=True)