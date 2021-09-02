import argparse
import pandas
import numpy as np
from pathlib import Path
from tensorflow import keras
from explainability import clustering_ef
from two_D_resnet import get_data


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with still images.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save the cluster labels in")
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-n', '--max_n_clusters', default=100, type=int,
                        help="Maximum number of clusters to be evaluated.")
    parser.add_argument('-mp', '--model_path', required=True)
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
    volumes['EF'] = volumes['Volume']  # just to allow reuse of cluster function
    actual_volumes_esv = volumes[volumes['ESV/EDV'] == 'ESV'].reset_index()
    actual_volumes_edv = volumes[volumes['ESV/EDV'] == 'EDV'].reset_index()

    # get esv train/validation/test data
    esv_train_still_images, _, _, \
        _, _, _, \
        esv_val_still_images, _, _ = get_data(
            args.input_directory, metadata_path, volume_type='ESV')
    esv_train_still_images.extend(esv_val_still_images)

    # get edv train/validation/test data
    edv_train_still_images, _, _, \
        _, _, _, \
        edv_val_still_images, _, _ = get_data(
            args.input_directory, metadata_path, volume_type='EDV')
    edv_train_still_images.extend(edv_val_still_images)

    print('Data loaded')

    predicted_volumes_esv = get_predictions(esv_train_still_images, args.model_path)
    predicted_volumes_edv = get_predictions(edv_train_still_images, args.model_path)
    actual_volumes_esv['EF'] = predicted_volumes_esv
    actual_volumes_edv['EF'] = predicted_volumes_edv

    clustering_ef.cluster_by_ef(actual_volumes_esv, args.max_n_clusters, output_directory, '_esv')
    clustering_ef.cluster_by_ef(actual_volumes_edv, args.max_n_clusters, output_directory, '_edv')


def get_predictions(still_images, model_path):
    predictions = []

    # load model
    print('Start loading model')
    model = keras.models.load_model(model_path)
    predicting_model = keras.Model(inputs=[model.input],
                                   outputs=model.layers[len(model.layers) - 1].output)
    print('End loading model')

    for i in range(len(still_images)):
        print("instance", i)
        instance = np.expand_dims(still_images[i], axis=0)
        prediction = float(predicting_model(instance).numpy()[0][0])
        predictions.append(prediction)
    return predictions


if __name__ == '__main__':
    main()
