from expose_deep_phonemizer_module import expose_dp
from Loader.cv_loader import Loader

expose_dp()

if __name__ == "__main__":
    loader = Loader(it_fraction=0.15, es_fraction=0.15)
    dataset = loader.load(30)

    print(f"Train size: {len(dataset['train'])}")
    for training_example in dataset["train"]:
        print(f"Sample: {training_example['sentence']}")
