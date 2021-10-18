import pandas as pd

from common.csv_utils import CsvReader
from common.io import get_dir, get_file_name
from common.input_tw import remove_special_symbol


def common_voice_metadata_iter(tsv_file_path):
    base_dir = get_dir(tsv_file_path)
    with CsvReader(tsv_file_path) as tsv:
        tsv.skip_header()
        reader = tsv.reader
        for row in reader:
            file = row[1]
            mp3_file = f"{file}"
            filename = get_file_name(mp3_file)
            wav_filepath = f"{base_dir}/clips/{filename}.wav"
            sentence = remove_special_symbol(row[2])
            yield wav_filepath, sentence


def get_all_common_voice_metadata_dataframe(base_dir):
    df = pd.DataFrame(columns=['wav_filepath', 'trans'])
    index = 0
    for wav_filepath, trans in common_voice_metadata_iter(f"{base_dir}/train.tsv"):
        # series = pd.Series({
        #     'wav_filepath': wav_filepath,
        #     'trans': trans
        # }, name=f"{index}")
        # df = df.append(series, ignore_index=False)
        df = df.append({
            'wav_filepath': wav_filepath,
            'trans': trans
        }, ignore_index=True)
        index += 1
    return df

