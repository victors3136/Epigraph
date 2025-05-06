import os
import torch
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from Loader.cv_loader import Loader

class WhisperFinetuner:
    def __init__(self, it_fraction: float, es_fraction: float, output_path: str, seed: int = 42):
        self.loader = Loader(it_fraction, es_fraction, seed=seed)
        self.output_path = output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "openai/whisper-small"
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def preprocess(self, batch):
        audio = batch["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]
        
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        input_values = inputs.input_features.squeeze(0)
        
        labels = self.processor.tokenizer(
            batch["sentence"],
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            "input_features": input_values,
            "labels": labels
        }

    def prepare_dataset(self, data_list):
        print("Preprocessing dataset for Whisper...")
        processed = [self.preprocess(sample) for sample in data_list]
        return Dataset.from_list(processed)

    def train(self, n_samples: int = 10_000, num_train_epochs: int = 3, batch_size: int = 8):
        print("Loading data...")
        data = self.loader.load(n_samples)
        
        train_dataset = self.prepare_dataset(data["train"])
        val_dataset = self.prepare_dataset(data["val"])
        
        args = TrainingArguments(
            output_dir=self.output_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            num_train_epochs=num_train_epochs,
            save_strategy="epoch",
            logging_dir=os.path.join(self.output_path, "logs"),
            fp16=to
