import librosa


def convert_flac_to_wav(flac_filepath, wav_filepath, resample_rate=16000):
    waveform, sample_rate = librosa.load(flac_filepath)
    waveform = librosa.resample(waveform, sample_rate, resample_rate)
    librosa.output.write_wav(wav_filepath, waveform, resample_rate)
