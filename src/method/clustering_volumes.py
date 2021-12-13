import argparse
import pandas
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from model.two_d_resnet import get_data
from utils import clustering as cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        default='../../data/still_images',
                        help='Directory with still images.')
    parser.add_argument('-o', '--output_directory',
                        help='Directory to save the cluster labels in.')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-e', '--prediction_error', default=13.57, type=float,
                        help='Prediction error of model.')
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'clustering_volume')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # get actual volumes of all still images
    metadata_path = Path(args.input_directory, args.frame_volumes_filename)
    file_list_data_frame = pandas.read_csv(metadata_path)
    volumes = file_list_data_frame[file_list_data_frame.Split == 'TRAIN'][['ImageFileName', 'Volume', 'ESV/EDV']]
    volumes = volumes.append(file_list_data_frame[file_list_data_frame.Split == 'VAL'][['ImageFileName', 'Volume', 'ESV/EDV']])
    volumes = volumes.reset_index()
    volumes['FileName'] = volumes['ImageFileName']
    actual_volumes_esv = volumes[volumes['ESV/EDV'] == 'ESV'].reset_index()
    actual_volumes_edv = volumes[volumes['ESV/EDV'] == 'EDV'].reset_index()

    # get ESV train/validation/test data
    esv_train_still_images, _, _, \
        _, _, _, \
        esv_val_still_images, _, _ = get_data(
            args.input_directory, metadata_path, volume_type='ESV')
    esv_train_still_images.extend(esv_val_still_images)

    # get EDV train/validation/test data
    edv_train_still_images, _, _, \
        _, _, _, \
        edv_val_still_images, _, _ = get_data(
            args.input_directory, metadata_path, volume_type='EDV')
    edv_train_still_images.extend(edv_val_still_images)

    print('Data loaded')

    predicted_volumes_esv = get_predictions(esv_train_still_images, args.model_path)
    predicted_volumes_edv = get_predictions(edv_train_still_images, args.model_path)
    actual_volumes_esv['Volume'] = predicted_volumes_esv
    actual_volumes_edv['Volume'] = predicted_volumes_edv

    cluster_by_volume(actual_volumes_esv, output_directory, '_esv', args.prediction_error)
    cluster_by_volume(actual_volumes_edv, output_directory, '_edv', args.prediction_error)


def get_predictions(still_images, model_path):
    """Get the volume values that the given model predicts for the still
    images."""
    predictions = []

    # load model
    model = keras.models.load_model(model_path)
    predicting_model = keras.Model(inputs=[model.input],
                                   outputs=model.layers[len(model.layers) - 1].output)

    for i in range(len(still_images)):
        instance = np.expand_dims(still_images[i], axis=0)
        prediction = float(predicting_model(instance).numpy()[0][0])
        predictions.append(prediction)
    return predictions


def cluster_by_volume(volumes, output_directory, file_ending='', error=13.57):
    """
    Employ Jenks Caspall algorithm to find interval borders for the volume
    range.
    The cluster or interval borders and cluster labels are saved to files.
    @param volumes: list of the volumes of all still images
    @param output_directory: directory for saving the interval borders and
    cluster labels
    @param file_ending: file ending that the files in the output directory
    should have, indicating whether the volumes are ESV or EDV
    @param error: average prediction error of predicting model
    """
    #
    natural_breaks = []
    data = []
    num_intervals = []
    avg_widths = []
    for n in range(2, len(volumes) - 1):
        num_intervals.append(n)
        natural_breaks, data = cl.jenks_caspall(volumes, 'Volume', n_clusters=n)
        width = 0
        num = 0
        for i in range(1, len(natural_breaks) - 1):
            width += (natural_breaks[i] - natural_breaks[i - 1])
            num += 1
        avg_width = width / num
        avg_widths.append(avg_width)
        print('Average width of intervals = ', avg_width)
        if avg_width < error:
            break

    df = pd.DataFrame({'num_intervals': num_intervals, 'avg_widths': avg_widths})
    df.to_csv(Path(output_directory, 'avg_widths' + file_ending + '.csv'), index=False)

    cluster_labels = data['Cluster'].tolist()

    with open(Path(output_directory, 'cluster_labels' + file_ending + '.txt'), 'w') as txt_file:
        for i in range(len(cluster_labels)):
            txt_file.write(str(cluster_labels[i]) + ' ' + str(volumes.at[i, 'Volume'])
                           + ' ' + str(volumes.at[i, 'FileName']) + '\n')

    with open(Path(output_directory, 'cluster_centers' + file_ending + '.txt'), 'w') as txt_file:
        for i in range(len(natural_breaks) - 1):
            txt_file.write(str(i) + ' ' + str([(natural_breaks[i] + natural_breaks[i + 1]) / 2.0]) + '\n')

    with open(Path(output_directory, 'cluster_upper_borders' + file_ending + '.txt'), 'w') as txt_file:
        for i in range(len(natural_breaks) - 1):
            txt_file.write(str(i) + ' ' + str([natural_breaks[i + 1]]) + '\n')


if __name__ == '__main__':
    main()
