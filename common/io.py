import glob
import shutil

import cv2
from colorama import Fore, Style
import os
import re


def info(msg):
    print(Fore.GREEN, end="")
    print(msg, end="")
    print(Style.RESET_ALL)


def info_error(msg):
    print(Fore.RED, end="")
    print(msg, end="")
    print(Style.RESET_ALL)


def is_file_exists(file_path):
    return os.path.isfile(file_path)


def move_file(source_file_path, destination_dir):
    # confirm_dirs(destination_dir)
    shutil.move(source_file_path, destination_dir)


def confirm_dirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def clear_folder(folder):
    files = glob.glob(f'{folder}/*.*')
    for file in files:
        try:
            os.unlink(file)
        except OSError as e:
            print("Error: %s : %s" % (file, e.strerror))


def remove_files(files):
    for file in files:
        try:
            os.unlink(file)
        except OSError as e:
            print("Error: %s : %s" % (file, e.strerror))


def get_file_list_by_pattern(folder, pattern):
    regex = re.compile(pattern)
    files = glob.glob(f"{folder}/*.*")
    result = []
    for file in files:
        match = regex.search(file)
        if match:
            result.append(file)
    return result


def read_text_file(file_path):
    with open(file_path, "r", encoding='UTF-8') as f:
        lines = f.readlines()
    return lines


def read_bytes_file(file_path):
    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        bytes = f.read(file_size)
    return bytes


def sed(file_path, reg_pattern):
    regex = re.compile(r'(\d+)s/(.+)/(.+)/')
    match = regex.search(reg_pattern)
    if not match:
        raise Exception("sid pattern error")

    '212s/255/18/'
    line_num = int(match.group(1)) - 1
    search_pattern = match.group(2)
    replace_txt = match.group(3)

    # info(f"sed {line_num} {search_pattern} {replace_txt}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    with open(file_path, "w") as f:
        line = lines[line_num]
        text_after = re.sub(search_pattern, replace_txt, line)
        lines[line_num] = text_after
        # for line in lines:
        #     f.write(re.sub(reg_pattern, txt, line))
        f.write("".join(lines))


def get_folder_list(the_dir):
    if not os.path.exists(the_dir):
        return []
    return [f"{the_dir}/{name}" for name in os.listdir(the_dir)
            if os.path.isdir(os.path.join(the_dir, name))
            ]


def get_file_extension(file):
    split_tup = os.path.splitext(file)
    return split_tup


def get_file_name(fullname):
    return os.path.basename(fullname).split('.')[0]


def get_dir_file_name(fullname):
    dir = os.path.dirname(fullname)
    name = get_file_name(fullname)
    return dir, name


def get_dir(fullname):
    return os.path.dirname(fullname)


def read_image(image_file_fullname):
    image = cv2.imread(image_file_fullname)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_lines_file(txt_file_path):
    with open(txt_file_path, 'r', encoding='UTF-8') as f:
        lines = f.read().split('\n')
    return lines