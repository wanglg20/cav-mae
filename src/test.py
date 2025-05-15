import torchaudio
filename = '/data/wanglinge/dataset/data/k700/val/audio/ju5Z13zmFlI_000114_000124.wav'

waveform, sr = torchaudio.load(filename)
print(waveform.shape)