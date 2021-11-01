import re
from datetime import timedelta
from typing import Generator


def get_srt_file_metadata_iter(srt_filepath) -> Generator[timedelta, timedelta, str]:
    with open(srt_filepath, "r", encoding='utf-8') as f:
        f_iter = iter(f)
        for line in f_iter:
            line = line.rstrip('\n')
            match, start_timedelta, end_timedelta = try_parse_srt_time_line(line)
            if match:
                caption = next(f_iter).rstrip('\n')
                yield start_timedelta, end_timedelta, caption


TIME_PATTERN = "([0-9]{2,2})\:([0-9]{2,2})\:([0-9]{2,2})\,([0-9]{3,3})"


def parse_time_str(text) -> timedelta:
    match = re.search(TIME_PATTERN, text)
    if not match:
        raise ValueError(f"can't parse '{text}' to timedelta.")
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))
    milliseconds = int(match.group(4))
    return timedelta(hours=hour, minutes=minute, seconds=second, milliseconds=milliseconds)


def group_pattern(pattern):
    pattern = pattern.replace('(', '')
    pattern = pattern.replace(')', '')
    return f"({pattern})"


SRT_TIME_LINE_PATTERN = f"{group_pattern(TIME_PATTERN)} --> {group_pattern(TIME_PATTERN)}"


def try_parse_srt_time_line(line) -> [bool, timedelta, timedelta]:
    match = re.search(SRT_TIME_LINE_PATTERN, line)
    if not match:
        return False, None, None
    start_time = match.group(1)
    end_time = match.group(2)
    return True, parse_time_str(start_time), parse_time_str(end_time)


def convert_timedelta_to_srt_time_format(t: timedelta) -> str:
    total_seconds = t.total_seconds()
    hour = int(total_seconds // 60 // 60)
    total_seconds -= int(hour * 60 * 60)
    minute = int(total_seconds // 60)
    total_seconds -= minute * 60
    s = str(total_seconds).split('.')
    second = int(s[0])
    milli_second = int(s[1]) if len(s) > 1 else 0
    return f"{hour:0>2d}:{minute:0>2d}:{second:0>2d},{milli_second:0>3d}"


def get_srt_timestamp_line(start_time: timedelta, end_time: timedelta) -> str:
    srt_line = f"{convert_timedelta_to_srt_time_format(start_time)}" \
               f" --> " \
               f"{convert_timedelta_to_srt_time_format(end_time)}"
    return srt_line


def get_srt_clip(start_time: timedelta, end_time: timedelta, caption: str):
    part_start = start_time - start_time
    part_end = end_time - start_time
    srt_line = get_srt_timestamp_line(part_start, part_end)
    srt_content = f"{srt_line}\n{caption}\n"
    return srt_content


def write_clip_srt_file(start_time, end_time, caption, target_srt_filepath):
    slice_srt_content = get_srt_clip(start_time, end_time, caption)
    with open(target_srt_filepath, "w", encoding='utf-8') as f:
        f.write(f"{slice_srt_content}")


