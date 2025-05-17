from datasets import load_dataset
from transformers.models.whisper import WhisperProcessor, \
                                        WhisperForConditionalGeneration
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

class WhisperFineTuner:
    def __init__(self, 
                 dataset_url: str, 
                 model_name: str = "openai/whisper-small", 
                 language: str = "romanian", 
                 task: str = "transcribe", 
                 target_freq: int = 16_000):
        print(f"Initializing fine tuner based on {dataset_url}...")
        self.dataset_url = dataset_url
        self.model_name = model_name
        self.language = language
        self.task = task
        self.target_freq = target_freq

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

    def ensure_target_frequency(self, audio):
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        if audio.get('sampling_rate') != self.target_freq:
            waveform = torchaudio.transforms.Resample(
                orig_freq=audio.get('sampling_rate'),
                new_freq=self.target_freq
            )(waveform)
        return waveform.numpy()

    def preprocess(self, batch):
        audio = self.ensure_target_frequency(batch["audio"])
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=self.target_freq
        ).input_features[0]
        labels = self.processor.tokenizer(batch["sentence"]).input_ids

        return {
            "input_features": input_features,
            "labels": labels
        }

    class Collator:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features):
            input_features = [torch.tensor(f["input_features"]) for f in features]
            label_ids = [f["labels"] for f in features]

            input_features = pad_sequence(input_features, batch_first=True)

            labels_batch = self.processor.tokenizer.pad(
                {"input_ids": label_ids},
                return_tensors="pt",
                padding=True
            )

            return {
                "input_features": input_features,
                "labels": labels_batch["input_ids"]
            }

    def load_and_prepare_data(self):
        print(f"Loading data...")
        dataset = load_dataset(self.dataset_url)
        dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset = dataset.map(self.preprocess, remove_columns=dataset["train"].column_names)
        dataset.set_format(type="torch")

        return dataset["train"], dataset["test"]

    def train(self, output_dir, batch_size=8, epochs=3, push_to_hub=False):
        print(f"Setting up training...")
        hub_kwargs = {
                "push_to_hub": True,
                "hub_model_id": output_dir,
                "hub_private_repo": False
            } if push_to_hub else {}

        train_dataset, eval_dataset = self.load_and_prepare_data()

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=20,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to="wandb",
            **hub_kwargs
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.tokenizer,
            data_collator=self.Collator(self.processor)
        )

        print(f"Training begins...")
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        if push_to_hub:
            print(f"Saving model...")
            trainer.push_to_hub(output_dir)