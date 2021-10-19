from video_processing.audio_to_video import invoke_ffmpeg


def convert_wmv_to_mp4(wmv_filepath, mp4_filepath):
    # ffmpeg_command = f"-i {wmv_filepath} -c:v libx264 -crf 23 -c:a aac -q:a 100 {mp4_filepath}"
    ffmpeg_command = f"-i {wmv_filepath} -c:v libx264 -c:a aac -b:a 160k {mp4_filepath}"
    return invoke_ffmpeg(ffmpeg_command)
