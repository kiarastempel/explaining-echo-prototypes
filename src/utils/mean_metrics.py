import tensorflow as tf
from tensorflow import keras


def distinct_mean(validation_dataset):
    for batch_index in range(:
        batch_targets = self.validation_data[1][batch_index]
        batch_samples = self.validation_data[0][batch_index]
        number_of_frames = len(samples.video)
        number_of_subvideos = int(number_of_frames / self.number_input_frames)
        prediction = []
        for _ in range(number_of_subvideos):

    mae = tf.metrics.MeanAbsoluteError(video.true_value, prediction)
    mean_value = tf.reduce_mean(predictions)


    return mae


def overlapping_mean(validation_dataset):
    return 0


