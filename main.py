# MusicGen test — generates music from a text prompt
# Model: facebook/musicgen-small
# This is a learning experiment, not a full project
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

model_id = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(model_id)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
inputs = processor(
    text=["love happy beat, slow melody,slow 808, 160bpm"],
    padding=True,
    return_tensors="pt",
).to(device)
audio_values = model.generate(**inputs, max_new_tokens=512)

sampling_rate = model.config.audio_encoder.sampling_rate


output_data = audio_values[0, 0].cpu().numpy()


scipy.io.wavfile.write("dark_trap_beat.wav", rate=sampling_rate, data=output_data)

print("Successfully saved to dark_trap_beat.wav!")
