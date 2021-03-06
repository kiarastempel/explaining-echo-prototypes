import pandas as pd
import numpy as np
import ast
from similarity import normalize_polygon, rotate_polygon, angles_to_centroid


class Image:
    def __init__(self, features, volume, file_name, image=None, segmentation=None,
                 normalized_rotations=[], feature_ranks=[], error_ranks=[],
                 average_rank_distance=None,
                 average_positive_rank_distance=None,
                 average_negative_rank_distance=None):
        self.features = features
        self.volume = volume
        self.file_name = file_name
        self.image = image
        # segmentation: {'X': list of coordinates, 'Y': list of coordinates}
        self.segmentation = segmentation
        self.normalized_rotations = normalized_rotations
        self.feature_ranks = feature_ranks
        self.error_ranks = error_ranks
        self.average_rank_distance = average_rank_distance
        self.average_positive_rank_distance = average_positive_rank_distance
        self.average_negative_rank_distance = average_negative_rank_distance


def read_cluster_labels(cluster_file):
    cluster_labels = []
    volumes = []
    file_names = []
    with open(cluster_file, 'r') as txt_file:
        for line in txt_file:
            line_split = line.split(' ')
            cluster_labels.append(int(line_split[0]))

            volumes.append(float(line_split[1]))
            file_names.append(line_split[2].rsplit()[0])
    return cluster_labels, volumes, file_names


def read_extracted_features(file_path):
    extracted_features = []
    with open(file_path, 'r') as txt_file:
        for line in txt_file:
            if not line.startswith('tf.Tensor'):
                if line.startswith('[['):
                    image_features = []
                    line = line.strip('[')
                if line.__contains__('shape'):
                    line = line.split(']')[0]
                    image_features.extend([float(v) for v in line.split()])
                    extracted_features.append(image_features)
                else:
                    image_features.extend([float(v) for v in line.split()])
    return extracted_features


def read_image_clusters(cluster_labels_file, image_features_file):
    # list of clusters
    # where each cluster is a list of its corresponding still images
    cluster_features = []

    image_cluster_labels, volume_file, file_names = \
        read_cluster_labels(cluster_labels_file)
    num_clusters = max(image_cluster_labels) + 1
    print(str(cluster_labels_file), ' num clusters: ', num_clusters)
    for i in range(num_clusters):
        cluster_features.append([])

    image_features = read_extracted_features(image_features_file)
    for i in range(len(image_features)):
        cluster_features[image_cluster_labels[i]].append(
            Image(image_features[i], volume_file[i], file_names[i]))
    return cluster_features


def read_image_cluster_centers(centers_file_path, image_known=True):
    cluster_centers = []
    new_center = True
    if image_known:
        i = 0
    else:
        i = -2
    with open(centers_file_path, 'r') as txt_file:
        for line in txt_file:
            line_split = line.split()
            if new_center:
                new_center = False
                if image_known:
                    volume = line_split[1]
                    file_name = line_split[2]
                else:
                    volume = None
                    file_name = None
                if len(line_split[3 + i].strip('[')) == 0:
                    features = []
                else:
                    features = [float(line_split[3 + i].strip('['))]
                for f in line_split[4 + i:]:
                    features.append(float(f))
            else:
                end = len(line_split) - 1
                if line_split[end].endswith(']'):
                    line_split[end] = line_split[end].strip(']')
                    if len(line_split[end]) == 0:
                        line_split.pop()
                    for f in line_split:
                        features.append(float(f))
                    cluster_centers.append(Image(features, volume, file_name))
                    new_center = True
                else:
                    for f in line_split:
                        features.append(float(f))
    return cluster_centers


def read_volume_cluster_centers(centers_file_path):
    cluster_centers = []
    with open(centers_file_path, 'r') as txt_file:
        for line in txt_file:
            line_split = line.split()
            volume = line_split[1].strip('[]')
            cluster_centers.append(float(volume))
    return cluster_centers


def read_prototypes(centers_file_path, volume_tracings_file_path=None,
                    actual_volumes=True):
    prototypes = {}
    new_center = True
    volume_cluster_index = 0
    prototypes[volume_cluster_index] = []
    with open(centers_file_path, 'r') as txt_file:
        for line in txt_file:
            line_split = line.split()
            if new_center:
                new_center = False
                if int(line_split[0]) is not volume_cluster_index:
                    volume_cluster_index = int(line_split[0])
                    prototypes[volume_cluster_index] = []
                if not line_split[2] == 'None':
                    volume = float(line_split[2])
                else:
                    volume = 0
                if not line_split[3] == 'None':
                    file_name = line_split[3]
                else:
                    file_name = None
                if len(line_split[4].strip('[')) == 0:
                    features = []
                else:
                    features = [float(line_split[4].strip('['))]
                for f in line_split[5:]:
                    features.append(float(f))
            else:
                end = len(line_split) - 1
                if line_split[end].endswith(']'):
                    line_split[end] = line_split[end].strip(']')
                    if len(line_split[end]) == 0:
                        line_split.pop()
                    for f in line_split:
                        features.append(float(f))
                    prototypes[volume_cluster_index].append(
                        Image(features, volume, file_name)
                    )
                    new_center = True
                else:
                    for f in line_split:
                        features.append(float(f))
    if volume_tracings_file_path and actual_volumes:
        volume_tracings_data_frame = pd.read_csv(volume_tracings_file_path)
        for volume_cluster_prototypes in prototypes.values():
            for prototype in volume_cluster_prototypes:
                volume = volume_tracings_data_frame.loc[
                    volume_tracings_data_frame['ImageFileName'] == prototype.file_name]['Volume'].iloc[0]
                prototype.volume = volume
    return prototypes


def get_segmentation_coordinates_of_prototypes(prototypes,
                                               volume_tracings_dict):
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            file_name = prototypes[i][j].file_name
            prototypes[i][j].segmentation = {
                'X': ast.literal_eval(volume_tracings_dict[file_name]['X']),
                'Y': ast.literal_eval(volume_tracings_dict[file_name]['Y'])}
    return prototypes


def get_normalized_rotations_of_prototypes_with_angles(prototypes,
                                                       rotation_extent,
                                                       num_rotations):
    for i in range(len(prototypes)):
        for j in range(len(prototypes[i])):
            print('file', prototypes[i][j].file_name)
            rotated_features = []
            points = [list(x) for x in zip(prototypes[i][j].segmentation['X'],
                                           prototypes[i][j].segmentation['Y'])]
            normalized_points = normalize_polygon(np.array(points))
            center = np.array(np.mean(normalized_points, axis=0))

            for angle in np.linspace(-rotation_extent, rotation_extent, num=num_rotations, endpoint=True):
                rotated_points = list(normalize_polygon(rotate_polygon(normalized_points, center, angle)))
                rotated_angles = list(angles_to_centroid(rotated_points, center))
                features = [
                    [rotated_points[i][0], rotated_points[i][1],
                     rotated_angles[i]] for i in range(len(points))
                ]
                rotated_features.append(features)
            prototypes[i][j].normalized_rotations = rotated_features
    return prototypes
