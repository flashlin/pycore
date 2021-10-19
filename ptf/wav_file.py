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

