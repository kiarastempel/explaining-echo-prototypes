import argparse
import os
import math
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import explainability.read_helpers as rh
from pathlib import Path
from dtw import *
from explainability.visualise_prototype_quality import a_d_amount, a_d_interval
from prototypes_quality import normalize_polygon, rotate_polygon, angles_to_centroid


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with still images.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save prototypes and evaluations in")
    parser.add_argument('-p', '--chosen_prototypes_file', default='test_chosen_prototypes.csv',
                        help='Name of file containing chosen prototypes')
    parser.add_argument('-re', '--rotation_extent', type=float, default=np.pi/8)
    parser.add_argument('-nr', '--num_rotations', type=int, default=9)
    parser.add_argument('-fv', '--frame_volumes_filename',
                        default='FrameVolumes.csv',
                        help='Name of the file containing frame volumes.')
    parser.add_argument('-vt', '--volume_type', default='ESV',
                        help='ESV, EDV or None')
    args = parser.parse_args()

    if args.output_directory is None:
        output_directory = Path(args.input_directory, 'results')
    else:
        output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_volumes_path = Path(args.input_directory, args.frame_volumes_filename)
    chosen_prototypes_path = Path(args.output_directory, args.chosen_prototypes_file)

    # get prototypes and corresponding segmentation coordinates
    volume_tracings_data_frame = pd.read_csv(frame_volumes_path)
    volume_tracings_data_frame = volume_tracings_data_frame.set_index('ImageFileName')
    volume_tracings_dict = volume_tracings_data_frame.to_dict(orient='index')

    # get most similar prototypes for instances
    chosen_prototypes_frame = pd.read_csv(chosen_prototypes_path)

    # go on: applicability domain plotten!
    a_d_interval([chosen_prototypes_frame['euclidean_diff_features']], [chosen_prototypes_frame['euclidean_diff_angles']], ['features - shape'],
                 num_prototypes=0,
                 similarity_measure='Euclidean Distance', diffs_type='features',
                 num_cut_offs=1000, label_size=12, ticks_size=12,
                 ylabel="Average shape distance")

    a_d_amount([chosen_prototypes_frame['euclidean_diff_angles']], [chosen_prototypes_frame['euclidean_diff_features']], ['features - shape'],
                 num_prototypes=120,
                 similarity_measure='Euclidean Distance', diffs_type='shapes',
                 num_cut_offs=1000, label_size=12, ticks_size=12,
                 ylabel="Average feature distance")

    plt.scatter(chosen_prototypes_frame['euclidean_diff_angles'], chosen_prototypes_frame['euclidean_diff_features'])
    plt.show()

    # iterate over all instances in chosen_prototypes file and each time:
    # show segmentation instance, segmentation of chosen (most similar)
    # prototype, and calculated distances (of volumes, features, shapes)
    i = 0
    for _, row in chosen_prototypes_frame.T.iteritems():
        i += 1
        instance = row['file_name']
        prototype = row['euclidean_prototype']
        shape_distance = row['euclidean_diff_angles']
        feature_distance = row['euclidean_diff_features']
        instance_volume = row['actual_volume']
        prototype_volume = row['euclidean_volume']
        if shape_distance < 2:
            continue

        # get segmentation of current instance
        instance_points = [list(x) for x in zip(
            ast.literal_eval(volume_tracings_dict[instance]['X']),
            ast.literal_eval(volume_tracings_dict[instance]['Y'])
        )]
        normalized_instance_points = normalize_polygon(instance_points)
        instance_center = np.array(np.mean(normalized_instance_points, axis=0))
        instance_angles = list(angles_to_centroid(normalized_instance_points, instance_center))
        instance_features = [
            [normalized_instance_points[i][0],
             normalized_instance_points[i][1],
             instance_angles[i]] for i in range(len(instance_points))]

        # get segmentation of most similar prototype rotation
        prototype_points = [list(x) for x in zip(
            ast.literal_eval(volume_tracings_dict[prototype]['X']),
            ast.literal_eval(volume_tracings_dict[prototype]['Y'])
        )]
        normalized_prototype_points = normalize_polygon(prototype_points)

        center = np.array(np.mean(normalized_prototype_points, axis=0))
        min_dist = sys.float_info.max
        min_angle = 0
        for angle in np.linspace(-args.rotation_extent, args.rotation_extent,
                                 num=args.num_rotations, endpoint=True):
            rotated_points = list(normalize_polygon(rotate_polygon(normalized_prototype_points, center, angle)))
            rotated_angles = list(angles_to_centroid(rotated_points, center))
            features = [
                [rotated_points[i][0], rotated_points[i][1],
                 rotated_angles[i]] for i in range(len(prototype_points))
            ]
            dist = dtw(features, instance_features).distance
            if dist < min_dist:
                min_dist = dist
                min_angle = angle

        instance_segmentation_x = [p[0] for p in normalized_instance_points]
        instance_segmentation_y = [p[1] for p in normalized_instance_points]
        plt.plot(instance_segmentation_x + instance_segmentation_x[0:1],
                 instance_segmentation_y + instance_segmentation_y[0:1],
                 label='Volume of still image instance ' + str(instance_volume))
        rotated_prototype_points = list(normalize_polygon(rotate_polygon(normalized_prototype_points, center, min_angle)))
        prototype_segmentation_x = [p[0] for p in rotated_prototype_points]
        prototype_segmentation_y = [p[1] for p in rotated_prototype_points]
        plt.plot(prototype_segmentation_x + prototype_segmentation_x[0:1],
                 prototype_segmentation_y + prototype_segmentation_y[0:1],
                 label='Volume of prototype ' + str(prototype_volume))
        plt.legend()
        plt.title('Shape distance: '
                  + str(int((min_dist * 100)) / 100.0)
                  + ', angle of prototype rotation: '
                  + str(int((min_angle * 100)) / 100.0)
                  + ', feature distance: '
                  + str(int((feature_distance * 100)) / 100.0))
        plt.show()


if __name__ == '__main__':
    main()

