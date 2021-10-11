import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_array_feature(value):
    """
    value_type: np.float32
    value: [1.0, 2.0]
    """
    # isinstance(value, list)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def ToFloat32ArrayFeature(value):
    return _bytes_feature(value.astype(np.float32).tostring())


def ToInt32ArrayFeature(value):
    return _bytes_feature(value.astype(np.int32).tostring())


def _parse_ndarray_feature(value):
    '''
    def parse_fn(example_proto):
        features_desc = {
            "audio": tf.FixedLenFeature((), tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features_desc)
        float32Array = tf.decode_raw(parsed_features['audio'], tf.float32)
    '''
    return tf.decode_raw(value, tf.float32)
