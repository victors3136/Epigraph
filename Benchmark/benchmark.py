import os
from datasets import load_dataset
from jiwer import wer, cer
from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.generation.configuration_utils import GenerationConfig
import torchaudio
import torch
from tqdm import tqdm

MODEL_PATHS = []
LANGUAGE = "ro"
SPLIT = "test"
MAX_SAMPLES = 1_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Common Voice Romanian test data...")
cv = load_dataset("mozilla-foundation/common_voice_11_0", LANGUAGE, split=SPLIT)
cv = cv.filter(lambda x: x["sentence"] is not None and x["audio"] is not None)
cv = cv.select(range(min(MAX_SAMPLES, len(cv))))

def speech_file_to_array_fn(batch):
    speech_array, _ = torchaudio.load(batch["audio"]["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["target_text"] = batch["sentence"].lower()
    return batch

cv = cv.map(speech_file_to_array_fn)

results = []
for model_id in MODEL_PATHS:
    fails = 0
    total = 0

    print(f"\nEvaluating {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id) \
                                     .to(DEVICE)

    predictions = []
    references = []

    for sample in tqdm(cv):
        input_features = processor(
            sample["speech"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)
        with torch.no_grad():
            try:
                predicted_ids = model.generate(
                    input_features,
                    language="ro",
                    task="transcribe",
                )
            except IndexError as ie:
                fails += 1
                continue
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].lower()

        predictions.append(transcription)
        references.append(sample["target_text"])

    wer_score = wer(references, predictions) * 100
    cer_score = cer(references, predictions) * 100
    print(f"Total samples: {len(cv)}, of which {fails} failed due to indexing errors")
    print(f"{model_id}: WER = {wer_score:.2f}%, CER = {cer_score:.2f}%")
    results.append((model_id, wer_score, cer_score))


results.sort(key=lambda x: (
    x[1], # sort by wer by default
    x[2]) # use cer as tiebreaker
)

print("\nFinal Benchmark Results:")
print(f"{'Model':<60} {'WER (%)':<10} {'CER (%)':<10}")
for model_id, wer_score, cer_score in results:
    print(f"{model_id:<60} {wer_score:<10.2f} {cer_score:<10.2f}")
