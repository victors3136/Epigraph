DATASET_CONFIGS = [
    {"it_fraction": 0.00, "es_fraction": 0.00, "ro_count": 10},
    {"it_fraction": 0.30, "es_fraction": 0.00, "ro_count": 10},
    {"it_fraction": 0.00, "es_fraction": 0.30, "ro_count": 10},
    {"it_fraction": 0.15, "es_fraction": 0.15, "ro_count": 10},
]

from expose_deep_phonemizer_module import expose_dp

expose_dp()

from Loader.cv_loader import Loader

for config in DATASET_CONFIGS:
    it, es, ro = config["it_fraction"], config["es_fraction"], config["ro_count"]
    name = f"{ro}-it{it}-es{es}"
    total_sample_ct = ro + int(it * ro) + int(es * ro)
    print(f"Creating {name} with IT: {it}, ES: {es}")
    loader = Loader(it_fraction=it, es_fraction=es)

    dataset = loader.load(n_samples=ro)
    output_dir = f"./cached_{name}"
    print(f"Saving {name} to {output_dir}")
    dataset.save_to_disk(output_dir)

    dataset.push_to_hub(f"victors3136/{name}", private=True)
