from pathlib import Path
import subprocess
import platform

import librosa
from matplotlib import pyplot as plt
from pydub import AudioSegment

from common.io import get_dir, get_file_name, get_dir_file_name


def read_wav_data(wav_filepath):
    return librosa.load(wav_filepath)


def get_ffmpeg_executable_filepath():
    if platform.system() == 'Windows':
        ffmpeg_path_bytes = subprocess.check_output("where ffmpeg", shell=True)  # returns bytes
    elif platform.system() == 'Linux':
        ffmpeg_path_bytes = subprocess.check_output("which ffmpeg", shell=True)
    ffmpeg_executable_path = ffmpeg_path_bytes.decode().strip()
    return ffmpeg_executable_path


def invoke_ffmpeg(command):
    ffmpeg_executable_path = get_ffmpeg_executable_filepath()
    print("ffmpeg_executable_path: ", ffmpeg_executable_path)
    cmd_command = f"{ffmpeg_executable_path} {command}"
    returned_value = subprocess.call(
        cmd_command, shell=True
    )  # returns the exit code in unix
    print("returned value:", returned_value)
    return returned_value


def convert_wav_to_mp4(mp3_filepath, mp4_filepath, image_filepath=None):
    if image_filepath is None:
        dir = get_dir(mp4_filepath)
        filename = get_file_name(mp4_filepath)
        image_filepath = f"{dir}/{filename}.png"
        waveform, sample_rate = read_wav_data(mp3_filepath)
        save_waveform_to_image(waveform, sample_rate, image_filepath)

    # ffmpeg_command = f"-loop 1 -i {image_filepath} -i {mp3_filepath} -c:a copy -c:v libx264 -shortest {mp4_filepath}"
    ffmpeg_command = f"-loop 1 -i {image_filepath} -i {mp3_filepath} -c:a aac -b:a 160k -c:v libx264 -shortest {mp4_filepath}"
    return invoke_ffmpeg(ffmpeg_command)


def save_waveform_to_image(waveform, sample_rate, image_filepath, text=''):
    """
    Args:
        waveform: Input signal
        sample_rate: Sampling rate of x
        text: Text to print
    """
    # print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, sample_rate, waveform.shape, waveform.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(waveform, color='gray')
    plt.title(f'Original waveform sr={sample_rate}')
    plt.xlim([0, waveform.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    # plt.show()
    # import IPython.display as ipd
    # ipd.display(ipd.Audio(data=waveform, rate=sample_rate))
    plt.savefig(image_filepath)


