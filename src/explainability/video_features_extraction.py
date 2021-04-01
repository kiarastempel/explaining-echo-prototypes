import argparse
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
from data_loader import tf_record_loader
import read_helpers as rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with the TFRecord files.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save the video features in")
    parser.add_argument('-f', '--number_input_frames', default=50, type=int)
    parser.add_argument('-cl', '--ef_clusters_file',
                        default='../../data/clustering_ef/cluster_labels_ef.txt',
                        help='Path to file containing ef cluster labels')
    parser.add_argument('-mp', '--model_path', required=True)
    parser.add_argument('-l', '--hidden_layer_index', default=14, type=int)
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    if output_directory is None:
        output_directory = Path(args.input_directory, 'video_features')
    output_directory.mkdir(parents=True, exist_ok=True)

    # get data of ef clustering
    ef_cluster_labels, actual_efs, file_names = rh.read_cluster_labels(args.ef_clusters_file)

    # get tf_records of train and validation data
    data_folder = Path(args.input_directory)
    train_record_file_names = data_folder / 'train' / 'train_*.tfrecord.gzip'
    validation_record_file_names = data_folder / 'validation' / 'validation_*.tfrecord.gzip'
    train_dataset = tf_record_loader.build_dataset(
        file_names=str(train_record_file_names),
        batch_size=1,
        shuffle_size=None,
        number_of_input_frames=args.number_input_frames)
    validation_dataset = tf_record_loader.build_dataset(
        file_names=str(validation_record_file_names),
        batch_size=1,
        shuffle_size=None,
        number_of_input_frames=args.number_input_frames)
    train_dataset = train_dataset.concatenate(validation_dataset)

    extract_features(args.model_path, args.hidden_layer_index,
                     ef_cluster_labels, train_dataset, args.number_input_frames,
                     output_directory)


def extract_features(model_path, hidden_layer_index, ef_cluster_labels,
                     train_dataset, number_input_frames, output_directory):
    # load model
    print("Start loading model")
    model = keras.models.load_model(model_path)
    print("End loading model")
    # keras.utils.plot_model(model.layers[0], to_file="data/resnet_18.png",
    #                        show_shapes=True)

    # extract features of videos of each ef-cluster at given hidden layer index
    print("Layer index: ", hidden_layer_index)
    num_ef_clusters = max(ef_cluster_labels) + 1
    print("Calculated " + str(num_ef_clusters) + " ef clusters")
    for i in range(num_ef_clusters):
        print("Start ef cluster " + str(i))
        extracted_features, ef = get_hidden_layer_features(
            model, train_dataset, ef_cluster_labels, i, hidden_layer_index,
            number_input_frames)

        # write features to file
        out_file = Path(output_directory, 'extracted_video_features_' + str(i) + ".txt")
        with open(out_file, "w") as txt_file:
            for f in range(len(extracted_features)):
                txt_file.write(str(extracted_features[f]) + "\n")


def get_hidden_layer_features(model, train_dataset, cluster_labels,
                              ef_cluster_index, layer_index,
                              number_input_frames):
    """
    Extract features at given layer_index of given model for all
    training instances belonging to cluster with ef_cluster_index.
    @return: list of features
    """
    extractor = keras.Model(inputs=model.get_layer(index=0).input,
                            outputs=model.layers[0].layers[layer_index].output)
    extracted_features = []
    ef = []
    i = 0
    for video, y in train_dataset:
        if cluster_labels[i] == ef_cluster_index:
            # print(i, ": ", y)
            first_frames = video[:, :number_input_frames, :, :, :]
            features = extractor(first_frames)
            extracted_features.append(features)
            ef.append(y[0])
        i += 1
    return extracted_features, ef


if __name__ == '__main__':
    main()
