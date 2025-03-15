# pip install transformers torch

import torch
from transformers import pipeline

# ...existing code...
whisper = pipeline("automatic-speech-recognition", "openai/whisper-large-v3")
transcription = whisper("./test.mp3", return_timestamps=True)
print(transcription)


print(transcription["text"])