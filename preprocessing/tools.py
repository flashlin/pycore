import numpy as np
from sklearn.preprocessing import MinMaxScaler


def min_max_scaler_array(an_array, minimum=0, maximum=1):
    an_array = np.reshape(an_array, (-1, 1))
    sc = MinMaxScaler(feature_range=(minimum, maximum))
    an_array = sc.fit_transform(an_array)
    an_array = np.reshape(an_array, -1)
    return an_array