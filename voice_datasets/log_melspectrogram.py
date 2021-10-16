import tensorflow as tf
from tensorflow.keras import layers

F_MIN = 0
F_MAX = 8000
SAMPLE_RATE = 16000
EPS = 1e-12

class LogMelgramLayer(tf.keras.layers.Layer):
    def __init__(self, num_fft=255, hop_length=128, mel_bins=40, **kwargs):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length

        N_SPECTROGRAM_BINS = num_fft // 2 + 1

        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=mel_bins,
            num_spectrogram_bins=N_SPECTROGRAM_BINS,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=F_MIN,
            upper_edge_hertz=F_MAX,
        )

        self.lin_to_mel_matrix = lin_to_mel_matrix

    def build(self, input_shape):
        self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        # tf.signal.stft seems to be applied along the last axis
        stfts = tf.signal.stft(
            input, frame_length=self.num_fft, frame_step=self.hop_length
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0])
        log_melgrams = _tf_log10(melgrams + EPS)
        return tf.expand_dims(log_melgrams, 3)

    def get_config(self):
        config = {'num_fft': self.num_fft, 'hop_length': self.hop_length}
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


"""
    Demo Sample:
    
    input_shape = (44100 * 10,)  # 10-sec mono audio input
    inputs = layers.Input(shape=input_shape, name='audio_waveform')

    log_melgram_layer = LogMelgramLayer(
        num_fft=NUM_FFT,
        hop_length=HOP_LENGTH,
    )

    log_melgrams = log_melgram_layer(inputs)

    some_network = get_your_network()
    outputs = some_network(log_melgrams)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
"""

