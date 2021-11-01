import numpy as np
from PIL.ImageFont import FreeTypeFont
from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import ImageFont, ImageDraw, Image
from moviepy.video.VideoClip import VideoClip

from common.io import get_file_name
from video_processing.srt_file import get_srt_file_metadata_iter


def draw_rectangle(image_frame, shape):
    [(x, y), (width, height)] = shape
    rectangle_shape = [(x, y), (x + width, y + height)]
    image_frame.rectangle(rectangle_shape, fill=(0, 0, 0))


def generate_image_frame(size):
    mode = "RGB"
    back_color = (0, 0, 0)
    img = Image.new(mode, size, back_color)
    image_frame = ImageDraw.Draw(img)
    return img, image_frame


def generate_caption_video_frame(caption: str, font: FreeTypeFont, size=(1280, 50)):
    def _compute_caption_bound():
        width, height = size
        caption_size = font.getsize(caption)
        caption_width, caption_height = caption_size
        caption_x = width / 2 - caption_width / 2
        caption_y = height - caption_height - 5
        return [(caption_x, caption_y), caption_size]

    img, image_frame = generate_image_frame(size)
    [caption_xy, caption_size] = _compute_caption_bound()
    image_frame.text(caption_xy, caption, font=font, fill=(0, 255, 134))
    image_frame = np.array(img)
    return image_frame


def generate_caption_video_frames_iter(caption: str, seconds_duration: float,
                                       font: FreeTypeFont, size=(1280, 50), fps=15):
    show_frames = round(fps * seconds_duration)
    for n in range(0, show_frames):
        image_frame = generate_caption_video_frame(caption, font, size)
        yield image_frame


def generate_caption_video_clip(caption: str, seconds_duration: float,
                                font: FreeTypeFont, size=(1280, 50)):
    def make_frame(t):
        image_frame = generate_caption_video_frame(caption, font, size)
        return image_frame
    video_clip = VideoClip(make_frame, duration=seconds_duration)
    return video_clip


def generate_empty_caption_video_clip(seconds_duration: float, size=(1280, 50)):
    def make_frame(t):
        img, image_frame = generate_image_frame(size)
        image_frame = np.array(img)
        return image_frame
    video_clip = VideoClip(make_frame, duration=seconds_duration)
    return video_clip


def generate_caption_video(caption: str, seconds_duration: float, video_filepath: str,
                           font: FreeTypeFont, size=(1280, 50), fps=15):
    for image_frame in generate_caption_video_frames_iter(caption, seconds_duration, font, size, fps):
        fourcc = VideoWriter_fourcc(*'XVID')  # MP42 XVID MP4V
        video = VideoWriter(f"{video_filepath}",
                            fourcc,
                            float(fps),
                            size)
        video.write(image_frame)
        video.release()


class SrtVideoGenerator:
    width = 1280
    height = 720
    fps = 15
    font_size = 30
    font_file = "./assets/NotoSansTC-Regular.otf"

    def __init__(self):
        self.font = ImageFont.truetype(self.font_file, self.font_size, encoding='utf-8')
        self.height = 50

    def _compute_caption_bound(self, caption):
        caption_size = self.font.getsize(caption)
        caption_width, caption_height = caption_size
        caption_x = self.width / 2 - caption_width / 2
        caption_y = self.height - caption_height - 5
        return [(caption_x, caption_y), caption_size]

    def _draw_caption(self, image_frame, caption):
        [caption_xy, caption_size] = self._compute_caption_bound(caption)
        # shape = [caption_xy, caption_size]
        # draw_rectangle(image_frame, shape)
        image_frame.text(caption_xy, caption, font=self.font, fill=(0, 255, 134))

    def generate_srt_clips(self, srt_filepath, out_dir, video_name=None):
        if video_name is None:
            video_name = get_file_name(srt_filepath)
        fourcc = VideoWriter_fourcc(*'XVID')  # MP42 XVID MP4V
        idx = 0
        for (start, end, caption) in get_srt_file_metadata_iter(srt_filepath):
            seconds_duration = (end - start).total_seconds()
            video = VideoWriter(f"{out_dir}/{video_name}-{idx:>04}.avi",
                                fourcc,
                                float(self.fps),
                                (self.width, self.height))
            self.draw_caption_in_video_frames(video, caption, seconds_duration)
            video.release()
            idx += 1

    def draw_caption_in_video_frames(self, video, caption, seconds_duration):
        fps = 15
        show_frames = round(fps * seconds_duration)
        mode = "RGB"
        back_color = (0, 0, 0)
        for n in range(0, show_frames):
            img = Image.new(mode, (self.width, self.height), back_color)
            image_frame = ImageDraw.Draw(img)
            self._draw_caption(image_frame, caption)
            image_frame = np.array(img)
            video.write(image_frame)
