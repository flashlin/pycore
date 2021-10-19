import time

import webrtcvad
import pyaudio
import threading
import numpy as np
from queue import Queue


def list_microphones(pyaudio_instance):
    info = pyaudio_instance.get_host_api_info_by_index(0)
    num_devices = info.get_all_common_voice_data('deviceCount')
    result = []
    for i in range(0, num_devices):
        if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get_all_common_voice_data('maxInputChannels')) > 0:
            name = pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get_all_common_voice_data('name')
            result += [[i, name]]
    return result


def get_input_device_id(device_name, microphones):
    for device in microphones:
        if device_name in device[1]:
            return device[0]


class LiveMicrophone:
    opening = False

    def __init__(self):
        self.voice_queue = Queue()

    def open(self, device_name="default"):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)
        RECORD_SECONDS = 50

        microphones = list_microphones(audio)
        selected_input_device_id = get_input_device_id(device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        frames = b''
        print("microphone open")
        self.opening = True
        while self.opening:
            frame = stream.read(CHUNK)
            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    self.voice_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def stop(self):
        # self.exit_event.set()
        self.opening = False
        self.voice_queue.put("close")
        print("microphone stopped")


# exit_event = threading.Event()
def start_live_microphone(fn_process):
    microphone = LiveMicrophone()

    def _open():
        microphone.open()

    def _read():
        while True:
            audio_frames = microphone.voice_queue.get()
            if audio_frames == "close":
                break
            float64_buffer = np.frombuffer(audio_frames, dtype=np.int16) / 32767
            fn_process(float64_buffer)
        pass
    listen_process = threading.Thread(target=_open, args=())
    listen_process.start()
    read_process = threading.Thread(target=_read, args=())
    read_process.start()
    return microphone
