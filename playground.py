DATASET_CONFIGS = [
    {"it_fraction": 0.35, "es_fraction": 0.15, "ro_count": 4_000},
    {"it_fraction": 0.15, "es_fraction": 0.35, "ro_count": 4_000},
]

from expose_deep_phonemizer_module import expose_dp

expose_dp()

from Loader.cv_loader import Loader
import time 
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

    retries = 3
    for i in range(retries):
        try:
            dataset.push_to_hub(f"victors3136/{name}", private=True)
            break
        except Exception as e:
            print(f"Push failed ({i+1}/{retries}): {e}")
            time.sleep(5)
