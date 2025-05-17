import os
import torch
import torchaudio
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
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        original_sr = audio["sampling_rate"]

        if original_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=16000)
            waveform = resampler(waveform)

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
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

    def collate_fn(self, batch):
        input_features = [torch.tensor(x["input_features"]) if not isinstance(x["input_features"], torch.Tensor) \
                          else x["input_features"] for x in batch]
        input_features = torch.stack(input_features)
    
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"]) if not isinstance(x["labels"], torch.Tensor) else x["labels"] for x in batch],
            batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
    
        return {"input_features": input_features, "labels": labels}


    def train(self, n_samples: int = 10_000, num_train_epochs: int = 3, batch_size: int = 8):
        print("Loading data...")
        data = self.loader.load(n_samples)
        
        train_dataset = self.prepare_dataset(data["train"])
        val_dataset = self.prepare_dataset(data["val"])
        
        args = TrainingArguments(
            output_dir=self.output_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            save_strategy="epoch",
            logging_dir=os.path.join(self.output_path, "logs"),
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
            push_to_hub=False
        )


        print("Starting training...")
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            data_collator=self.collate_fn,
        )

        trainer.train()
        self.model.save_pretrained(self.output_path)
        self.processor.save_pretrained(self.output_path)
        print(f"Model saved to {self.output_path}")
