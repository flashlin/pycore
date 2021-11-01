import pandas as pd

from common.csv_utils import CsvReader
from common.input_tw import remove_special_symbol
from common.io import get_dir


def my_voice_metadata_iter(wav_csv_filepath):
    base_dir = get_dir(wav_csv_filepath)
    with CsvReader(wav_csv_filepath) as tsv:
        tsv.skip_header()
        reader = tsv.reader
        for row in reader:
            filename = row[0]
            wav_filepath = f"{base_dir}/{filename}.wav"
            sentence = remove_special_symbol(row[1])
            yield wav_filepath, sentence


def get_my_voice_metadata_dataframe(wav_csv_filepath, fn_filter=None):
    df = pd.DataFrame(columns=['wav_filepath', 'trans'])
    index = 0
    for wav_filepath, trans in my_voice_metadata_iter(wav_csv_filepath):
        if fn_filter is not None:
            if not fn_filter(wav_filepath, trans):
                continue
        df = df.append({
            'wav_filepath': wav_filepath,
            'trans': trans
        }, ignore_index=True)
        index += 1
    return df
