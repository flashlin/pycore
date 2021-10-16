import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


class CsvReader:
    def __init__(self, csv_file):
        self._csv_file = csv_file

    def __enter__(self):
        self._filestream = open(self._csv_file, encoding='utf-8')
        self.reader = csv.reader(self._filestream, delimiter='\t')
        return self

    def __exit__(self, type, value, traceback):
        self._filestream.close()

    def skip_header(self):
        next(self.reader)


class CsvWriter:
    def __init__(self, csv_file, delimiter='\t'):
        self._csv_file = csv_file
        self._delimiter = delimiter

    def __enter__(self):
        self._filestream = open(self._csv_file, "w", encoding='utf-8', newline='')
        self.writer = csv.writer(self._filestream, delimiter=self._delimiter)
        return self

    def __exit__(self, type, value, traceback):
        self._filestream.close()

    def write(self, row):
        self.writer.writerow(row)


def read_csv(csv_file_path):
    # col_names = pd.read_csv(csv_file_path, nrows=0, sep='\t', lineterminator='\r').columns
    # types_dict = {'A': int, 'B': float}
    # types_dict.update({col: str for col in col_names if col not in types_dict})
    return pd.read_csv(csv_file_path, sep='\t')


def target_encode_multiclass(x, y):
    y = y.astype(str)
    category_dict = {}
    for category in y:
        category_dict[category] = category
    enc = ce.OneHotEncoder().fit(y)
    y_onehot = enc.transform(y)
    class_names = y_onehot.columns
    print(y_onehot)
    x_obj = x.select_dtypes('object')
    x = x.select_dtypes(exclude='object')
    for class_name in class_names:
        print(f"class_name={class_name}")
        enc = ce.TargetEncoder()
        enc.fit(x_obj, y_onehot[class_name])
        temp = enc.transform(x_obj)
        temp.columns = [f"{column_name}_{class_name}" for column_name in temp.columns]
        x = pd.concat([x, temp], axis=1)  # add to original dataset
    return x


def label_encode(csv_data_frame, category_columns):
    for category_name in category_columns:
        pre_encode_data_frame = csv_data_frame[category_name]
        encoder = LabelEncoder().fit(pre_encode_data_frame)
        encoded_data = encoder.transform(pre_encode_data_frame)
        # encoded_data_frame.columns = [ f"{category_name}_encoded" ]
        encoded_data_frame = pd.DataFrame(encoded_data, columns=[f"{category_name}_encoded"])
        # train.drop(columns=['name', 'text'], inplace=True)
        csv_data_frame = pd.concat([csv_data_frame, encoded_data_frame], axis=1)
    return csv_data_frame


def label_encode2(csv_data_frame, category_columns):
    pre_encode_data_frame = csv_data_frame[category_columns]
    encoder = LabelEncoder().fit(pre_encode_data_frame)
    encoded_data = encoder.transform(pre_encode_data_frame)
    encoded_category_names = [f"{name}_encoded" for name in category_columns]
    encoded_data_frame = pd.DataFrame(encoded_data, columns=encoded_category_names)
    # train.drop(columns=['name', 'text'], inplace=True)
    return pd.concat([csv_data_frame, encoded_data_frame], axis=1)
