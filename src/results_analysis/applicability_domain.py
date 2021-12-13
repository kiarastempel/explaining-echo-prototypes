import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import trapz


def applicability_domain_by_interval(differences, prediction_errors, labels,
                                     num_contained_prototypes=0,
                                     similarity_measure='Euclidean Distance',
                                     diffs_type='Features',
                                     num_cut_offs=1000,
                                     label_size=12, ticks_size=12,
                                     ylabel='Average prediction error'):
    """Plot average prediction error of 'best' (i.e. highest cosine similarity)
    instances, i.e. instances with cosine similarity lying in interval [0, x]
    (x = x axis)."""
    # sort errors by corresponding ascending distance to prototype
    sorted_diffs, sorted_errors = sort_errors_by_diffs(differences,
                                                       prediction_errors,
                                                       similarity_measure,
                                                       num_contained_prototypes)
    # get mean prediction error for each segment of instances/cut off point
    cut_off_points = []
    if 'Similarity' in similarity_measure:
        for d in sorted_diffs:
            cut_off_points.append(np.linspace(max(d), min(d), num_cut_offs,
                                              endpoint=False))  # 1, 0
    else:
        for d in sorted_diffs:
            cut_off_points.append(np.linspace(min(d), max(d), num_cut_offs,
                                              endpoint=True))  # 0, 1
    cut_off_amounts = [[] for _ in range(len(differences))]

    mean_errors_cut = []
    for i in range(len(sorted_diffs)):
        mean_errors_cut.append([])
        cut_off_amounts.append([])
        j = 0
        old_j = 0
        error_sum = 0
        for cut_off in cut_off_points[i]:
            if 'Similarity' in similarity_measure:
                while j < len(sorted_errors[i]) and sorted_diffs[i][j] >= cut_off:
                    error_sum += sorted_errors[i][j]
                    j += 1
                mean_errors_cut[i].append(error_sum / j)
                print(i, ':', j - old_j, ':', error_sum / j, cut_off)
                cut_off_amounts[i].append(j)
                old_j = j
            else:
                while j < len(sorted_errors[i]) and sorted_diffs[i][j] <= cut_off:
                    error_sum += sorted_errors[i][j]
                    j += 1
                mean_errors_cut[i].append(error_sum / j)
                cut_off_amounts[i].append(j)
                old_j = j

    # plot using x = cut_offs, y = mean_diffs_cut
    for i in range(len(labels)):
        plt.plot(cut_off_points[i], mean_errors_cut[i], label=labels[i])  # , markersize='1'
    if 'Similarity' in similarity_measure:
        plt.xlim(max(cut_off_points[0]), min(cut_off_points[0]))
    plt.xlabel('Applicability Domain (' + similarity_measure + ' of ' + diffs_type + ')', fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.title('Applicability Domain')
    plt.legend()
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.show()

    # calculate area under curve
    xlim = cut_off_points[0][-1]
    for i in range(len(labels)):
        if cut_off_points[i][-1] < xlim:
            xlim = cut_off_points[i][-1]
    for i in range(len(labels)):
        x_dist = abs(cut_off_points[i][0] - cut_off_points[i][1])
        mean_errors = []
        for j in range(len(mean_errors_cut[i])):
            if cut_off_points[i][j] < xlim:
                mean_errors.append(mean_errors_cut[i][j])
        area = trapz(mean_errors, dx=x_dist)
        print('Area under curve', labels[i], ':', '%.2f' % area)


def applicability_domain_by_amount(differences, prediction_errors, labels,
                                   num_contained_prototypes=0,
                                   similarity_measure='Euclidean Distance',
                                   diffs_type='Features',
                                   num_cut_offs=1000,
                                   label_size=13, ticks_size=15,
                                   ylabel='Average prediction error',
                                   window_size=15):
    """Plot average prediction error of 'best' (i.e. closest prototype)
    n (= x axis) instances."""
    # sort errors by corresponding ascending distance to prototype
    sorted_diffs, sorted_errors = sort_errors_by_diffs(differences,
                                                       prediction_errors,
                                                       similarity_measure,
                                                       num_contained_prototypes)
    # get mean prediction error for each segment of instances/cut off point
    cut_off_points = np.linspace(0, 1, num_cut_offs, endpoint=True)[1:]
    cut_off_amounts = []

    for cut_off in cut_off_points:
        cut_index = int(len(sorted_diffs[0]) * cut_off)
        cut_off_amounts.append(cut_index)

    mean_errors_cut = []
    for i in range(len(sorted_diffs)):
        mean_errors_cut.append([])
        for cut_index in cut_off_amounts:
            mean_errors_cut[i].append(sum([sorted_errors[i][j] for j in range(
                cut_index)]) / cut_index)

    save_applicability_domain(cut_off_amounts, mean_errors_cut[0], labels[0],
                              '../../data/ad.csv')
    running_mean(window_size, cut_off_amounts, mean_errors_cut, labels)

    # plot using x = cut_offs, y = mean_diffs_cut
    for i in range(len(labels)):
        plt.plot(cut_off_amounts, mean_errors_cut[i], label=labels[i])  # , markersize='1'
    plt.xlabel('Applicability Domain (' + similarity_measure + ' of '
               + diffs_type + '): #instances, instances sorted from best to '
                              'worst prototype distance', fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.title('Applicability Domain')
    plt.legend()
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.show()


def sort_errors_by_diffs(differences, prediction_errors, similarity_measure,
                         num_contained_prototypes):
    """Sort prediction errors by corresponding ascending distance to
    prototype."""
    diffs = []
    for d in differences:
        diffs.append(np.array(d))
    # first index: highest ('best') similarity
    sorted_index_diffs = []
    for d in diffs:
        if 'Similarity' in similarity_measure:
            # similarity: sort descending
            sorted_index_diffs.append(np.argsort(d)[::-1])
        else:
            # distance: sort ascending
            sorted_index_diffs.append(np.argsort(d))
    sorted_errors = []
    sorted_diffs = []
    for i in range(len(diffs)):
        sorted_errors.append([])
        sorted_diffs.append([])
        for index in sorted_index_diffs[i]:
            sorted_errors[i].append(prediction_errors[i][index])
            sorted_diffs[i].append(diffs[i][index])
        sorted_errors[i] = sorted_errors[i][num_contained_prototypes:]
        sorted_diffs[i] = sorted_diffs[i][num_contained_prototypes:]
    return sorted_diffs, sorted_errors


def save_applicability_domain(cut_off_amounts, mean_errors_cut, labels, path):
    """Save the x/y-pairs of the applicability domain to a csv file."""
    df = pd.DataFrame({'cut_off': cut_off_amounts,
                       labels + '_error': mean_errors_cut
                       })
    df.to_csv(path)


def running_mean(window_size, cut_off_amounts, mean_errors_cut, labels):
    """Calculate running mean by calculating the mean for a sliding window with
    given window size. Every window_size-th mean is saved or printed."""
    # save
    for i in range(len(labels)):
        means = []
        for j in range(int(len(cut_off_amounts[0:200]) / window_size) - 1):
            mean = 0
            for h in range(window_size):
                mean += mean_errors_cut[i][j * window_size + h]
            mean /= window_size
            means.append(mean)
        print('Means of', labels[i], ':')
        for m in means:
            print(round(m, 2))