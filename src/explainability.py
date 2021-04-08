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
import matplotlib.animation as animation
from numpy.random import default_rng


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
    attributions = []
    rng = default_rng(5)
    for i in range(5):
        baselines = rng.integers(low=0, high=255, size=samples.shape)
        integrated_gradients_algorithm = IntegratedGradients(model, layer=None, method="gausslegendre", n_steps=50,
                                                             internal_batch_size=16)
        explanation = integrated_gradients_algorithm.explain(samples, baselines=baselines, target=predictions)
        attributions.append(explanation.attributions)

    mean_attributions = np.mean(attributions, axis=0)
    mean_attributions = mean_attributions.squeeze(axis=0)
    for video_pos, video in enumerate(mean_attributions):
        attr = np.moveaxis(video, 0, -1)
        original_image = samples[video_pos][0]
        original_image = original_image[:, :, None]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(100, 50))
        ax[0].imshow(original_image.squeeze(), cmap='gray')
        ax[0].set_title('Original Image')
        visualize_image_attr(attr=attr, original_image=original_image, method='blended_heat_map',
                             sign='all', show_colorbar=True, title='Overlaid Attributions',
                             plt_fig_axis=(fig, ax[1]), use_pyplot=True, cmap='viridis')

        fig = plt.figure()
        frames = []
        mini = np.min(video)
        maxi = np.max(video)
        for frame in video:
            frames.append([plt.imshow(frame, cmap='viridis', vmin=mini, vmax=maxi, animated=True)])

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                        repeat_delay=1000)
        # ani.save('movie.mp4')
        plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    args = input_arguments.get_test_arguments()
    calculate_integrated_gradients(args.number_input_frames, args.dataset, args.model_path, args.input_directory,
                                   args.target,
                                   args.batch_size, args.resolution)
