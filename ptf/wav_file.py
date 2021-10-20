import tensorflow as tf
import tensorflow_io as tfio


def read_wav_data(wav_filepath, resample_rate=16000):
    file = tf.io.read_file(wav_filepath)
    audio, sample_rate = tf.audio.decode_wav(file, desired_channels=1)

    # if sample_rate != resample_rate:
    #     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    #     audio = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=resample_rate)
    #     sample_rate = resample_rate

    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    return audio, sample_rate


def waveform_to_spectrogram(waveform, frame_length=256, frame_step=160, fft_length=384):
    """
        frame_length = 256 An integer scalar Tensor. The window length in samples.
        frame_step = 160 An integer scalar Tensor. The number of samples to step.
        fft_length = 384 An integer scalar Tensor. The size of the FFT to apply.
            If not provided, uses the smallest power of 2 enclosing frame_length.
    """
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return spectrogram