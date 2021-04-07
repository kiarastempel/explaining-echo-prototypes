import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import input_arguments
from data_loader import tf_record_loader
from alibi.explainers import IntegratedGradients
from alibi.utils.visualization import visualize_image_attr
import matplotlib.pyplot as plt


def calculate_integrated_gradients(number_input_frames, dataset, model_path, input_directory, target, batch_size,
                                   resolution):
    random_seed = 5
    tf.random.set_seed(random_seed)
    data_folder = Path(input_directory)
    test_record_file_names = data_folder / 'test' / 'test_*.tfrecord.gzip'
    test_dataset = tf_record_loader.build_dataset(str(test_record_file_names), batch_size,
                                                  None, number_input_frames, resolution,
                                                  False, dataset, target, False)
    one_batch_dataset = test_dataset.take(1)
    samples = np.concatenate([x for x, y in one_batch_dataset], axis=0)

    # subclassing model in tensorflow breaks some features, so we have to define input and output explicit
    model_ = keras.models.load_model(model_path)
    inputs = keras.Input(shape=(number_input_frames, samples.shape[2], samples.shape[3]), dtype=tf.float32)
    outputs = model_(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    predictions = model.predict(one_batch_dataset)
    baseline = np.ones_like(samples[0])
    baseline *= 255
    integrated_gradients_algorithm = IntegratedGradients(model, layer=None, method="gausslegendre", n_steps=5,
                                                         internal_batch_size=16)
    explanation = integrated_gradients_algorithm.explain(samples, baselines=None, target=predictions)
    attributions = explanation.attributions

    for video_pos, video in enumerate(attributions[0]):
        for row in range(len(video)):
            attr = video[row]
            original_image = samples[video_pos][row]
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            visualize_image_attr(attr=None, original_image=original_image, method='original_image',
                                 title='Original Image', plt_fig_axis=(fig, ax[0]), use_pyplot=False)

            visualize_image_attr(attr=attr.squeeze(), original_image=original_image, method='blended_heat_map',
                                 sign='all', show_colorbar=True, title='Overlaid Attributions',
                                 plt_fig_axis=(fig, ax[1]), use_pyplot=True)


if __name__ == "__main__":
    # execute only if run as a script
    args = input_arguments.get_test_arguments()
    calculate_integrated_gradients(args.number_input_frames, args.dataset, args.model_path, args.input_directory,
                                   args.target,
                                   args.batch_size, args.resolution)
