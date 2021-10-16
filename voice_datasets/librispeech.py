from glob import glob

import pandas as pd

from common.io import get_dir


def librispeech_metadata_iter(txt_filepath):
    base_dir = get_dir(txt_filepath)
    # print(f"librispeech_metadata_iter='{txt_filepath}'")
    with open(txt_filepath, "r", encoding='utf-8') as f:
        for line in iter(f):
            ss = line.split(' ')
            filename = ss[0]
            trans = ss[1]
            wav_filepath = f"{base_dir}/{filename}.wav"
            yield wav_filepath, trans


def all_librisppech_metadata_iter(base_dir):
    for txt in glob(f"{base_dir}/**/*.txt", recursive=True):
        for wav_filepath, trans in librispeech_metadata_iter(txt):
            yield wav_filepath, trans


def get_all_librispeech_metadata_dataframe(base_dir):
    df = pd.DataFrame(columns=['wav_filepath', 'trans'])
    for wav_filepath, trans in all_librisppech_metadata_iter(base_dir):
        df = df.append({
            'wav_filepath': wav_filepath,
            'trans': trans.lower()
        }, ignore_index=True)
    return df
