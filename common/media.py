from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips, AudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from common.io import get_file_extension, get_dir_file_name


def merge_avi_audio(avi_file, audio_file, target_file):
    [_, ext] = get_file_extension(target_file)
    codec = "libx264"  # mpeg4
    if ext == ".avi":
        codec = "png"
    audio_clip = AudioFileClip(audio_file)
    avi_clip = VideoFileClip(avi_file)
    avi_clip = avi_clip.set_audio(audio_clip)
    video = CompositeVideoClip([avi_clip])
    video.write_videofile(target_file, codec=codec)
    video.close()


def read_caption_audio_file(caption_file):
    with open(caption_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    d = lines[0].split(' ')
    start_time = float(d[0])
    duration = float(d[1])
    t_start_end = lines[1].replace('\n', '').split(" --> ")
    caption = lines[2]
    return [start_time, duration, t_start_end[0], t_start_end[1], caption]


def get_audio_duration_str(duration):
    millis = duration * 1000
    millis = int(millis)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = (millis / (1000 * 60 * 60)) % 24
    hours = int(hours)
    millis -= hours * 1000 * 60 * 60 + minutes * 1000 * 60 + seconds * 1000
    return f"{hours}:{minutes}:{seconds:}.{millis}"


def make_quiet_audio_clip(duration):
    # make_frame_440 = lambda t: [sin(440 * 2 * pi * t)]
    return AudioClip(lambda t: [0], duration=duration, fps=44100)


def merge_avi_audio_clips(avi_file, audio_files, target_file):
    [_, ext] = get_file_extension(target_file)
    codec = "libx264"  # mpeg4
    if ext == ".avi":
        codec = "png"
    video_file_clip = VideoFileClip(avi_file)

    audio_clips = []
    audio_duration = 0
    for audio_file in audio_files:
        [audio_dir, audio_filename] = get_dir_file_name(audio_file)
        audio_name = get_file_extension(audio_filename)[0]
        [start_duration, duration, t_start, t_end, caption] = read_caption_audio_file(f"{audio_dir}/{audio_name}.txt")
        caption = caption.replace('\n', '')
        quiet_duration = start_duration - audio_duration
        if quiet_duration > 0:
            audio_clips.append(make_quiet_audio_clip(quiet_duration))
            audio_duration += quiet_duration
        audio_clip = AudioFileClip(audio_file)
        print(f"{t_start} --> {t_end} {caption}")
        audio_clips.append(audio_clip)
        audio_duration += duration

    final_audio = concatenate_audioclips(audio_clips)
    final_video = video_file_clip.set_audio(final_audio)
    final_video.write_videofile(target_file, codec=codec)


class VideoClipFrameIter:
    def __init__(self, clip: VideoFileClip):
        self.clip = clip
        self.current = -1
        self.max_frames = clip.reader.nframes
        self.iter = clip.iter_frames()
        self.every = int(clip.fps / 15)
        self.last_frame = None

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.increase_frames()
        if self.current < self.max_frames:
            return frame
        # return frame
        raise StopIteration

    def increase_frames(self):
        frame = self.last_frame
        for n in range(0, self.every):
            if self.current < self.max_frames - 1:
                frame = self.last_frame = next(self.iter)
                self.current += 1
        return frame


class SrtWriter:
    index = 0

    def __init__(self, file):
        self._file = file

    def get_timespan_str(self, duration):
        millis = duration * 1000
        millis = int(millis)
        seconds = (millis / 1000) % 60
        seconds = int(seconds)
        minutes = (millis / (1000 * 60)) % 60
        minutes = int(minutes)
        hours = (millis / (1000 * 60 * 60)) % 24
        hours = int(hours)
        millis -= hours * 1000 * 60 * 60 + minutes * 1000 * 60 + seconds * 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def write_caption(self, caption, start_duration, duration):
        self.index += 1
        end_duration = start_duration + duration
        with open(self._file, "a", encoding='utf-8') as f:
            line = f"{self.index}\n"
            time_span = f"{self.get_timespan_str(start_duration)} --> {self.get_timespan_str(end_duration)}"
            line += f"{time_span}\n"
            line += f"{caption}\n"
            line += "\n"
            f.write(line)
            f.flush()
        return time_span