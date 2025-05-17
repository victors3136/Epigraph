from Tuner.fine_tuner import WhisperFineTuner

if __name__ == "__main__":
    tuner = WhisperFineTuner("victors3136/dataset-5k-00it-00sp")
    tuner.train("victors3136/asr-model-5k-00it-00sp", epochs=1, push_to_hub=True)