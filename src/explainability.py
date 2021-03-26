import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import input_arguments
from data_loader import tf_record_loader
from alibi.explainers import IntegratedGradients
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
    integrated_gradients_algorithm = IntegratedGradients(model, layer=None, method="gausslegendre", n_steps=50,
                                                         internal_batch_size=16)
    explanation = integrated_gradients_algorithm.explain(samples, baselines=None, target=predictions)
    attributions = explanation.attributions

    cmap_bound = np.abs(attributions).max()
    for video_pos, video in enumerate(attributions[0]):
        for row in range(len(video)):
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(112, 112))
            # original images
            ax[0].imshow(samples[video_pos][row].squeeze(), cmap='gray')

            # attributions
            attr = video[row]
            im = ax[1].imshow(attr.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

            # positive attributions
            attr_pos = attr.clip(0, 1)
            im_pos = ax[2].imshow(attr_pos.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

            # negative attributions
            attr_neg = attr.clip(-1, 0)
            im_neg = ax[3].imshow(attr_neg.squeeze(), vmin=-cmap_bound, vmax=cmap_bound, cmap='PiYG')

            ax[0].set_title(f'Prediction: {predictions[video_pos]}')
            ax[1].set_title('Attributions')
            ax[2].set_title('Positive attributions')
            ax[3].set_title('Negative attributions')

            for ax in fig.axes:
                ax.axis('off')

            fig.colorbar(im, cax=fig.add_axes([0.93, 0.25, 0.03, 0.6]))
            plt.show()


if __name__ == "__main__":
    # execute only if run as a script
    args = input_arguments.get_test_arguments()
    calculate_integrated_gradients(args.number_input_frames, args.dataset, args.model_path, args.input_directory,
                                   args.target,
                                   args.batch_size, args.resolution)
