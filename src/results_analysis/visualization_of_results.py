import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import applicability_domain as ad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directories', nargs='+',
                        default=['../../data/results_esv_features',
                                 '../../data/results_edv_features'],
                        help='Directories containing results that should be'
                             'compared.')
    parser.add_argument('-l', '--legend_labels', nargs='+',
                        default=['Feature distance (ESV)',
                                 'Feature distance (EDV)'],
                        help='Labels for plots.')
    args = parser.parse_args()

    # analyse prototypes
    prototypes_data = pd.read_csv(
        Path(args.input_directories[0], 'prototypes_chosen_prototypes.csv'))
    plt.scatter(prototypes_data['actual_volume'],
                prototypes_data['prediction_error'],
                label='actual volume')
    plt.xlabel('Actual volume')
    plt.ylabel('Prediction Error')
    plt.title('Actual Volume related to Prediction Error of Prototypes')
    plt.show()
    prototype_file_names = prototypes_data['file_name'].to_list()
    num_prototypes = len(prototype_file_names)
    print('Number of prototypes', num_prototypes)
    print('Mean prediction error of 10 prototypes with biggest volume',
          prototypes_data.tail(10)['prediction_error'].mean(),
          '\n')

    # analyse how prototypes are distributed in train data
    prototypes_of_train_data = pd.read_csv(
        Path(args.input_directories[0],
             'train_chosen_prototypes.csv'))
    prototypes_of_train_data_sorted = prototypes_of_train_data.sort_values(
        'euclidean_diff_features', ignore_index=True)
    delete_indices = []
    for i, row in prototypes_of_train_data_sorted.T.iteritems():
        if row['file_name'] in prototype_file_names:
            print('Prototype at index', i,
                  'with error', row['prediction_error'],
                  'and diff', row['euclidean_diff_features'])
            delete_indices.append(i)
    print('Mean prediction error of 10 instances with closest prototype',
          prototypes_of_train_data_sorted.tail(100)['prediction_error'].mean(),
          '\n')

    results_1 = pd.read_csv(
        Path(args.input_directories[0], 'test_chosen_prototypes.csv'))
    results_2 = pd.read_csv(
        Path(args.input_directories[1], 'test_chosen_prototypes.csv'))
    results_lists = [results_1, results_2]

    # get prediction errors, distances of features and shapes, and weighted sums
    prediction_errors = get_column_of_all_csv(results_lists, 'prediction_error')
    diffs_euclidean_features = get_column_of_all_csv(results_lists, 'euclidean_diff_features', scale=True)
    diffs_euclidean_angles = get_column_of_all_csv(results_lists, 'euclidean_diff_angles', scale=True)
    diffs_euclidean_features_angles = []
    for i in range(len(diffs_euclidean_features)):
        linear_sums = []
        for j in range(len(diffs_euclidean_features[i])):
            linear_sums.append(0.5 * diffs_euclidean_features[i][j] + 0.5 * diffs_euclidean_angles[i][j])
        diffs_euclidean_features_angles.append(linear_sums)

    labels = args.legend_labels
    diffs = diffs_euclidean_features

    # visualize applicability domain
    ad.applicability_domain_by_amount(diffs, prediction_errors, labels, 0,
                                      similarity_measure='Euclidean Distance',
                                      diffs_type='Features')
    ad.applicability_domain_by_interval(diffs, prediction_errors, labels, 0,
                                        similarity_measure='Euclidean Distance',
                                        diffs_type='Features')

    # plots of distances to most similar prototype
    hist_of_diffs(diffs[0],
                  similarity_measure='Euclidean Distance',
                  diffs_type='features')
    multiple_boxplots(diffs, labels,
                      'Boxplots of Distances to most similar Prototype')

    # correlation of volume of still image and volume of its closest prototype
    scatter_plot(results_lists[0]['actual_volume'],
                 results_lists[0]['euclidean_volume'],
                 'Volume of Still Image',
                 'Volume of closest Prototype',
                 'Correlation of Volumes of Image and Volume of its closest '
                 'Prototype')

    # correlation of distance to closest prototype and prediction error
    plot_diffs_and_prediction_error_points(
        diffs[0],
        prediction_errors[0],
        similarity_measure='Euclidean Distance',
        diffs_type='features')

    # 2d histogram: x=diff and y=prediction error
    two_d_hist(diffs[0], prediction_errors[0],
               similarity_measure='Euclidean Distance',
               diffs_type='features')


def plot_diffs_and_prediction_error_points(
        diffs, prediction_error, similarity_measure='Cosine similarity',
        diffs_type='features'):
    plt.plot(diffs, prediction_error, 'o', markersize='2')
    plt.xlabel(similarity_measure + ' to closest prototype ' + diffs_type)
    plt.ylabel('Prediction error')
    plt.title('Correlation of Distance to Closest Prototype and Prediction '
              'error')
    plt.legend()
    plt.show()


def two_d_hist(diffs, prediction_error, similarity_measure='Cosine similarity',
               diffs_type='features', label_size=21, ticks_size=17):
    plt.hist2d(diffs, prediction_error, bins=(50, 50), cmap=plt.cm.jet)
    for i in range(len(diffs)):
        if diffs[i] < 0:
            print(diffs[i])
    plt.xlabel(similarity_measure + ' to closest prototype ' + diffs_type,
               fontsize=label_size)
    plt.ylabel('Prediction error', fontsize=label_size)
    plt.title('2D-Histogram showing Correlation of Distance to closest '
              'Prototype and Prediction Error')
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.show()


def hist_of_diffs(diffs, similarity_measure='Cosine similarity',
                  diffs_type='features'):
    plt.hist(diffs, bins=int(0.25 / 0.01))
    plt.plot([np.mean(diffs) for i in
              range(len(diffs))], label='mean distance')
    plt.xlabel(similarity_measure + ' of ' + diffs_type
               + ' to closest prototype')
    plt.ylabel('Number of Echocardiograms')
    plt.title('Histogram showing the Distribution of Distances to most similar'
              'Prototype (' + similarity_measure + ')')
    plt.legend()
    plt.show()


def plot_diffs(diffs, similarity_measure='Euclidean distance',
               diffs_type='features'):
    plt.plot(diffs, 'o')
    plt.plot([np.mean(diffs) for i in range(len(diffs))], label='mean')
    plt.xlabel('Index of echocardiogram')
    plt.ylabel(similarity_measure + ' of ' + diffs_type
               + ' to closest prototype')
    plt.legend()
    plt.show()


def multiple_boxplots(diffs, labels, title):
    # Multiple box plots on one axis
    fig, ax = plt.subplots()
    ax.boxplot(diffs, labels=labels)
    plt.title(title)
    plt.show()


def scatter_plot(x, y, x_label, y_label, title):
    plt.plot(x, y, 'o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def get_column_of_all_csv(csv_list, col, scale=False):
    columns = []
    scaler = MinMaxScaler()
    for csv_file in csv_list:
        column = csv_file[col]
        if scale:
            column = scaler.fit_transform(column.values.reshape(-1, 1)).flatten()
        columns.append(column)
    return columns


if __name__ == '__main__':
    main()
