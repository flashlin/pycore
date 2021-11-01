from PIL import ImageFont
from cv2 import VideoWriter_fourcc, VideoWriter
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from pydub import AudioSegment

from common.io import get_dir_file_name, get_file_name
from video_processing.srt_file import get_srt_file_metadata_iter, convert_timedelta_to_srt_time_format
from video_processing.srt_video import generate_caption_video, generate_caption_video_clip, \
    generate_empty_caption_video_clip


def generate_wav_srt_audio_iter(wav_filepath):
    base_dir, filename = get_dir_file_name(wav_filepath)
    srt_filepath = f"{base_dir}/{filename}.srt"
    # sound = AudioSegment.from_file(wav_filepath, "wav")
    sound = AudioFileClip(wav_filepath)
    srt_iter = get_srt_file_metadata_iter(srt_filepath)
    for start_time, end_time, caption in srt_iter:
        # start = start_time.total_seconds() * 1000
        # end = end_time.total_seconds() * 1000
        # audio_segment = sound[start:end]
        start = convert_timedelta_to_srt_time_format(start_time)
        end = convert_timedelta_to_srt_time_format(end_time)
        audio_segment = sound.subclip(start, end)
        # audio_segment.export("d:/demo/Downloads/1.wav", format="wav")
        # yield timedelta(milliseconds=start), timedelta(milliseconds=end), segment, caption
        yield start_time, end_time, caption, audio_segment


def generate_wav_srt_video_audio_clips_iter(wav_filepath, font, size=(1280, 50)):
    for start_time, end_time, caption, audio_segment in generate_wav_srt_audio_iter(wav_filepath):
        seconds_duration = (end_time - start_time).total_seconds()
        video_clip = generate_caption_video_clip(caption, seconds_duration, font, size)
        yield start_time, end_time, caption, audio_segment, video_clip


def generate_wav_all_srt_video_audio_clips_iter(wav_filepath, font, size=(1280, 50)):
    time = 0.0
    for start_time, end_time, caption, audio_clip in generate_wav_srt_audio_iter(wav_filepath):
        if time != start_time.total_seconds():
            duration = start_time.total_seconds() - time
            video_clip = generate_empty_caption_video_clip(duration, size)
            time = end_time.total_seconds()
            yield start_time, end_time, caption, None, video_clip
        duration = (end_time - start_time).total_seconds()
        video_clip = generate_caption_video_clip(caption, duration, font, size)
        yield start_time, end_time, caption, audio_clip, video_clip


def split_wav_to_video_by_srt(wav_filepath, out_dir=None, size=(1280, 50)):
    dir, filename = get_dir_file_name(wav_filepath)
    if out_dir is None:
        out_dir = dir
    font_file = "./assets/NotoSansTC-Regular.otf"
    font = ImageFont.truetype(font_file, size=30, encoding='utf-8')
    idx = 0
    for start_time, end_time, caption, audio_clip, video_clip in generate_wav_srt_video_audio_clips_iter(wav_filepath, font, size):
        duration = (end_time - start_time).total_seconds()
        print(f"{duration} '{caption}' {type(audio_clip)}")
        video_filepath = f"{out_dir}/{filename}-{idx:>04}.mp4"
        video_clip.audio = CompositeAudioClip([audio_clip])
        video_clip.write_videofile(f"{video_filepath}", fps=15)
        idx += 1


def wav_to_video_by_srt(wav_filepath, out_dir=None, size=(1280, 50), fps=15):
    base_dir, filename = get_dir_file_name(wav_filepath)
    if out_dir is None:
        out_dir = base_dir
    font_file = "./assets/NotoSansTC-Regular.otf"
    font = ImageFont.truetype(font_file, size=30, encoding='utf-8')

    output_filepath = f"{out_dir}/{filename}.mp4"
    # fourcc = VideoWriter_fourcc(*'XVID')
    # new_video = VideoWriter(f"{out_dir}/{filename}.mp4", fourcc, float(fps), size)
    all_clips = []
    idx = 0
    for start_time, end_time, caption, audio_clip, video_clip in generate_wav_all_srt_video_audio_clips_iter(
            wav_filepath, font, size):
        duration = (end_time - start_time).total_seconds()
        print(f"{idx=} {duration} '{caption}' {type(audio_clip)}")
        if audio_clip is not None:
            video_clip.audio = CompositeAudioClip([audio_clip])
        all_clips.append(video_clip)
        idx += 1
    final_clip = concatenate_videoclips(all_clips)
    # final_clip.to_videofile("output.mp4", fps=fps, remove_temp=False)
    final_clip.to_videofile(output_filepath, fps=fps)
    # new_video.release()
