import sys
import numpy as np
import cv2
import json
from tqdm import tqdm
from tensorflow import train


def extract_metadata(file_name, input_directory):
    video = cv2.VideoCapture(str(input_directory / 'Videos' / file_name))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_width, frame_height


def save_metadata(output_directory, frames_per_second, frame_width, frame_height, mean, std, number_of_test_samples,
                  number_of_train_samples, number_of_validation_samples):
    metadata = {'metadata': {
        'frames_per_second': frames_per_second,
        'frame_height': frame_height,
        'frame_width': frame_width,
        'number_of_test_samples': number_of_test_samples,
        'number_of_train_samples': number_of_train_samples,
        'number_of_validation_samples': number_of_validation_samples,
        'mean': mean,
        'std': std,
        'channels': 1
    }
    }

    with open(output_directory / 'tf_record' / 'metadata.json', 'w') as outfile:
        json.dump(metadata, outfile)


def load_video(file_name, needed_frames):
    video = cv2.VideoCapture(file_name)
    frame_list = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fps_relative = fps / needed_frames
    if needed_frames > int(frame_count * fps_relative):
        return None
    for i in range(int(frame_count * (needed_frames / fps))):
        frame_position = int(i * fps_relative)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_position-1)
        ret, frame = video.read()
        if frame.shape[2] > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_list.append(cv2.imencode('.jpeg', frame)[1].tostring())
    video.release()
    return frame_list


def calculate_train_mean_and_std(input_directory, file_names, standardisation_sample=None):
    n = 0
    mean = 0
    M2 = 0
    if standardisation_sample is None:
        standardisation_sample = len(file_names)
    filenames_to_use = file_names[0:standardisation_sample]
    for file_name in tqdm(filenames_to_use, file=sys.stdout):
        video = cv2.VideoCapture(str(input_directory / 'Videos' / file_name))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in range(frame_count):
            ret, frame = video.read()
            if frame.shape[2] > 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape
            for i in range(height):
                for j in range(width):
                    x = frame.item(i, j)
                    n = n + 1
                    delta = x - mean
                    mean = mean + delta / n
                    M2 = M2 + delta * (x - mean)
    var = M2 / (n - 1)
    return mean, np.sqrt(var)


def float_feature(value):
    """Returns a float_list from a float / double."""
    return train.Feature(float_list=train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return train.Feature(int64_list=train.Int64List(value=[value]))


def bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return train.Feature(bytes_list=train.BytesList(value=values))
