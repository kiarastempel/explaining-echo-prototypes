import argparse
import pandas as pd
import matplotlib.pyplot as plt
import read_helpers as rh
from statistics import mean, stdev
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory',
                        default='../../data/still_images',
                        help='Directory with still images.')
    parser.add_argument('-o', '--output_directories', nargs='+',
                        default=['../../data/results'],
                        help='Directory to save prototypes and evaluations in.')
    parser.add_argument('-l', '--legend_labels', nargs='+',
                        default=['1'],
                        help='Labels for plots.')
    parser.add_argument('-p', '--prototypes_filename', default='prototypes_esv.txt',
                        help='Name of file containing prototypes')
    parser.add_argument('-cp', '--chosen_prototypes_file', default='test_chosen_prototypes.csv',
                        help='Name of file containing chosen prototypes')
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-vt', '--volume_type', default='ESV',
                        help='ESV, EDV or None')
    args = parser.parse_args()

    frame_volumes_path = Path(args.input_directory, args.frame_volumes_filename)

    prototypes, ranks = rank_comparison(
        args.output_directories, args.chosen_prototypes_file,
        args.prototypes_filename, frame_volumes_path)

    plt.boxplot(ranks, labels=args.legend_labels)
    plt.ylabel(
        'Avg. Distance between feature-difference-rank and prediction-error-rank')
    plt.show()


def rank_comparison(output_directories, chosen_prototypes_file,
                    prototypes_filename, frame_volumes_path):
    ranks = []
    for output_directory in output_directories:
        # get most similar prototypes for instances
        chosen_prototypes_path = Path(output_directory, chosen_prototypes_file)
        chosen_prototypes_frame = pd.read_csv(chosen_prototypes_path)

        # sort instances according to feature distance to closest prototype
        # -> results in feature distance rank
        # sort instances according to prediction error
        # -> results in prediction error rank
        chosen_prototypes_frame['rank_euclidean_diff_features'] \
            = chosen_prototypes_frame['euclidean_diff_features'].rank(method='min')
        chosen_prototypes_frame['rank_euclidean_prediction_error'] \
            = chosen_prototypes_frame['prediction_error'].rank(method='min')

        # read prototypes
        prototypes = rh.read_prototypes(Path(output_directory, prototypes_filename), frame_volumes_path)

        # iterate over all instances
        # for each instance, add feature distance rank and prediction error rank
        # to corresponding prototype lists
        for i, row in chosen_prototypes_frame.T.iteritems():
            prototype = row['euclidean_prototype']
            feature_distance_rank = row['rank_euclidean_diff_features']
            prediction_error_rank = row['rank_euclidean_prediction_error']
            prototypes = update_prototype_rank(prototypes, prototype, feature_distance_rank, prediction_error_rank)

        # for each prototype: calculate average distance between both ranks
        # (if rank distance is greater than some threshold, delete prototype)
        prototypes = calculate_rank_distances(prototypes)
        all_average_rank_distances = []
        all_average_pos_rank_distances = []
        all_average_neg_rank_distances = []
        for i in range(len(prototypes)):
            for j in range(len(prototypes[i])):
                all_average_rank_distances.append(prototypes[i][j].average_rank_distance)
                if prototypes[i][j].average_positive_rank_distance:
                    all_average_pos_rank_distances.append(prototypes[i][j].average_positive_rank_distance)
                if prototypes[i][j].average_negative_rank_distance:
                    all_average_neg_rank_distances.append(prototypes[i][j].average_negative_rank_distance)
        print('- Mean/Std/Min/Max of all Ranks, only of positive ranks, '
              'only of negative ranks: ')
        print('Mean all:', mean(all_average_rank_distances))
        print('Mean pos:', mean(all_average_pos_rank_distances))
        print('Mean neg:', mean(all_average_neg_rank_distances))
        print('Std all:', stdev(all_average_rank_distances))
        print('Std pos:', stdev(all_average_pos_rank_distances))
        print('Std neg:', stdev(all_average_neg_rank_distances))
        print('Min all:', min(all_average_rank_distances))
        print('Min pos:', min(all_average_pos_rank_distances))
        print('Min neg:', min(all_average_neg_rank_distances))
        print('Max all:', max(all_average_rank_distances))
        print('Max pos:', max(all_average_pos_rank_distances))
        print('Max neg:', max(all_average_neg_rank_distances), '\n')
        ranks.append(all_average_rank_distances)
    return prototypes, ranks


def update_prototype_rank(prototypes, prototype_file_name, feature_rank, error_rank):
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            if prototype_file_name == prototypes[i][j].file_name:
                prototypes[i][j].feature_ranks = prototypes[i][j].feature_ranks + [feature_rank]
                prototypes[i][j].error_ranks = prototypes[i][j].error_ranks + [error_rank]
    return prototypes


def calculate_rank_distances(prototypes):
    del_indices = []
    sum_distances = 0
    sum_distances_positive = 0
    sum_distances_negative = 0
    num_ranks_positive = 0
    num_ranks_negative = 0
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            num_ranks = len(prototypes[i][j].feature_ranks)
            for k in range(num_ranks):
                distance = prototypes[i][j].feature_ranks[k] - prototypes[i][j].error_ranks[k]
                sum_distances += distance
                if distance < 0:
                    # not wanted: small feature distance,
                    # but large prediction error
                    sum_distances_negative += distance
                    num_ranks_negative += 1
                else:
                    # positive rank distance (i.e. large feature distance, small prediction error) is okay
                    sum_distances_positive += distance
                    num_ranks_positive += 1
            if num_ranks == 0:
                # delete prototype later if none of instances was assigned to it
                del_indices.append((i, j))
            else:
                prototypes[i][j].average_rank_distance = sum_distances / num_ranks
            if num_ranks_negative > 0:
                prototypes[i][j].average_negative_rank_distance = sum_distances_negative / num_ranks_negative
            if num_ranks_positive > 0:
                prototypes[i][j].average_positive_rank_distance = sum_distances_positive / num_ranks_positive
            sum_distances = 0
            sum_distances_positive = 0
            sum_distances_negative = 0
            num_ranks_positive = 0
            num_ranks_negative = 0
    del_indices.reverse()
    for to_delete in del_indices:
        del prototypes[to_delete[0]][to_delete[1]]
    return prototypes


if __name__ == '__main__':
    main()

