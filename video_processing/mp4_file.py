"""
pip3 install moviepy
"""
from datetime import timedelta
from moviepy.editor import VideoFileClip
from common.io import get_dir, get_file_name, info
from video_processing.srt_file import get_srt_clip, write_clip_srt_file, get_srt_file_metadata_iter


def write_slice_mp4_file(mp4_filepath: str, start_time: timedelta, end_time: timedelta, target_filepath: str):
    start_seconds = start_time.total_seconds()
    end_seconds = end_time.total_seconds()

    clip = VideoFileClip(mp4_filepath).subclip(start_seconds, end_seconds)
    clip.write_videofile(target_filepath)
    clip.close()


def write_video_to_mono_wav(clip_video, target_wav_filepath):
    clip_video.audio.write_audiofile(target_wav_filepath, fps=16000,
                                     codec='pcm_s16le',
                                     ffmpeg_params=['-ac', '1'])  # Convert to mono


def write_clip_wav_file(mp4_filepath: str, start_time: timedelta, end_time: timedelta, target_filepath: str):
    start_seconds = start_time.total_seconds()
    end_seconds = end_time.total_seconds()
    clip = VideoFileClip(mp4_filepath).subclip(start_seconds, end_seconds)
    write_video_to_mono_wav(clip, target_filepath)


def get_video_clip(video_file_clip, start_time, end_time):
    start_seconds = start_time.total_seconds()
    end_seconds = end_time.total_seconds()
    video_clip = video_file_clip.subclip(start_seconds, end_seconds)
    return video_clip


def get_mp4_to_clip_srt_iter(mp4_filepath: str):
    mp4_dir = get_dir(mp4_filepath)
    filename = get_file_name(mp4_filepath)
    srt_filepath = f"{mp4_dir}/{filename}.srt"
    clip_idx = 0
    for start_time, end_time, caption in get_srt_file_metadata_iter(srt_filepath):
        clip_name = f"{filename}-{clip_idx:0>4d}"
        # start_seconds = start_time.total_seconds()
        # end_seconds = end_time.total_seconds()
        # video_clip = VideoFileClip(mp4_filepath).subclip(start_seconds, end_seconds)
        video_clip = get_video_clip(VideoFileClip(mp4_filepath), start_time, end_time)
        srt_clip = get_srt_clip(start_time, end_time, caption)

        yield {
            'start_time': start_time,
            'end_time': end_time,
            'caption': caption,
            'clip_name': clip_name,
            'video_clip': video_clip,
            'srt_clip': srt_clip
        }
        clip_idx += 1


def extract_mp4_to_clip_srt_wav(mp4_filepath, target_dir):
    for clip in get_mp4_to_clip_srt_iter(mp4_filepath):
        clip_srt_filepath = f"{target_dir}/{clip['clip_name']}.srt"
        clip_wav_filepath = f"{target_dir}/{clip['clip_name']}.wav"
        info(f"write wav clip {clip_wav_filepath}")
        write_video_to_mono_wav(clip['video_clip'], clip_wav_filepath)
        info(f"write srt clip {clip_srt_filepath}")
        write_clip_srt_file(clip['start_time'], clip['end_time'], clip['caption'], clip_srt_filepath)