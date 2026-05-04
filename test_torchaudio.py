import torch
import torchaudio

# creating a test array of random numbers
sr = 48000
audio = torch.rand(1, 48000)
# try to resample from 48000 to 44100
resampled = torchaudio.functional.resample(audio, sr, 44100)
print(resampled.shape)
