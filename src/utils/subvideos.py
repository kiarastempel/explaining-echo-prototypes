import tensorflow as tf


def get_distinct_splits(video, number_of_frames):
    number_of_subvideos = int(video.shape[1] / number_of_frames)
    subvideos = []
    for i in range(number_of_subvideos):
        start = i * number_of_frames
        end = start + number_of_frames
        subvideos.append(video[:, start: end:, :, :, :])
    return tf.concat(subvideos, 0)


def get_overlapping_splits(video, number_of_frames):
    number_of_subvideos = int(video.shape[1] / (number_of_frames / 2)) - 1
    half_number_of_frames = int(number_of_frames / 2)
    subvideos = []

    for i in range(number_of_subvideos):
        start = i * half_number_of_frames
        end = start + number_of_frames
        subvideos.append(video[:, start: end:, :, :, :])
    return tf.concat(subvideos, 0)


def get_first_frames(video, number_of_frames):
    return video[:, :number_of_frames, :, :, :]
