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