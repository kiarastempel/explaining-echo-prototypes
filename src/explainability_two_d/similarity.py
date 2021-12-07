from __future__ import division  # to avoid integer devision problem
import math
import numpy as np
from shapely.geometry import Polygon
from dtw import *
from pathlib import Path
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, Normalizer
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def compare_polygons_with_lengths_and_angles(prototype, instance_points):
    prototype_points = [list(x) for x in zip(prototype.segmentation['X'],
                                             prototype.segmentation['Y'])]
    prototype_p = list(normalize_polygon(np.array(prototype_points)))
    prototype_angles = angles_to_adjacent_edge(prototype_p)
    prototype_edge_lengths = calculate_edges_lengths(prototype_p)
    prototype_features = list(zip(prototype_angles, prototype_edge_lengths))

    instance_p = list(normalize_polygon(np.array(instance_points)))
    instance_angles = angles_to_adjacent_edge(instance_p)
    instance_edge_lengths = calculate_edges_lengths(instance_p)
    instance_features = list(zip(instance_angles, instance_edge_lengths))

    # uncomment to use dtw for comparison (and comment out euclidean distance)
    # dist = dtw(prototype_features, instance_features).distance

    # uncomment to use dtw with rotating starting point for comparison
    # dist = compare_polygons_multiple_dtw(prototype_features, instance_features)

    # calculate euclidean distance for comparison
    dist = 0
    for i in range(len(instance_features)):
        dist = dist + euclidean(prototype_features[i], instance_features[i])
    return dist


# compare two polygons whereas first polygon is rotated (from -pi/8 to +pi/8)
# and each point of it is used as starting point for dtw once
def compare_polygons_rotation_translation_invariant(prototype, instance_points,
                                                    rotation_extent=np.pi / 8,
                                                    num_rotations=9):
    instance_p = normalize_polygon(np.array(instance_points))
    instance_center = np.array(np.mean(instance_p, axis=0))
    instance_angles = list(angles_to_centroid(instance_p, instance_center))
    instance_features = [
        [instance_p[i][0], instance_p[i][1], instance_angles[i]] for i in
        range(len(instance_points))]
    min_dist = dtw(prototype.normalized_rotations[0],
                   instance_features).distance

    angles = np.linspace(-rotation_extent, rotation_extent, num=num_rotations,
                         endpoint=True)
    for prototype_rotation_features in prototype.normalized_rotations:
        # uncomment to use dtw with rotating starting point
        # dist = compare_polygons_multiple_dtw(prototype_rotation_features, instance_features)

        # uncomment to use dtw for comparison
        # dist = dtw(prototype_rotation_features, instance_features).distance

        # use euclidean distance for comparison
        dist = 0
        for i in range(len(instance_features)):
            dist = dist + euclidean(prototype_rotation_features[i],
                                    instance_features[i])
        min_dist = min(min_dist, dist)
    return min_dist


# rotate points about center point by angle using rotation matrix
# first subtract center (translate), then rotate about (0,0) and
# add center again (untranslate)
def rotate_polygon(points, center, angle=np.pi / 4):
    rotated_points = np.dot(
        points - center,
        np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    ) + center
    return rotated_points


def normalize_polygon(points):
    # scale points (x,y) of polygon:
    # translate using polygon mean, i.e. translate such that center of polygon
    # is at (0,0), then normalize by dividing by std_y
    # x' = (x - mean_x) / std_y
    # y' = (y - mean_y) / std_y
    mean = np.mean(points, axis=0)
    std_y = np.std(points, axis=0)[0]
    points = (points - mean) / std_y
    return points


# compare lists of points of two polygons using dtw
# whereas each point of first polygon is used as starting point once
def compare_polygons_multiple_dtw(features_1, features_2):
    # based on polygon/HMM similarity paper
    min_align_distance = dtw(features_1, features_2,
                             keep_internals=False).distance
    min_index = 0
    for i in range(len(features_1)):
        features_1 = features_1[i:] + features_1[:i]
        alignment = dtw(features_1, features_2, keep_internals=False)
        min_align_distance = min(min_align_distance, alignment.distance)
        # if alignment.distance < min_align_distance:
        #     min_align_distance = alignment.distance
        #     min_index = i
    # alignment.plot(type="threeway")
    # print("Distance:", min_align_distance)
    # print("index", min_index)
    return min_align_distance


# compare angles of two polygons (just one by one)
def compare_angles(points_1, points_2):
    angles_1 = angles_to_adjacent_edge(points_1)
    angles_2 = angles_to_adjacent_edge(points_2)
    diff = 0
    for i in range(min(len(angles_1), len(angles_2))):
        diff = diff + abs(angles_1[i] - angles_2[i])
    return diff


# source for idea, slightly modified:
# https://stackoverflow.com/questions/30271926/python-algorithm-how-to-do-simple-geometry-shape-match
def angles_to_adjacent_edge(points):
    def vector(tail, head):
        return tuple(h - t for h, t in zip(head, tail))

    points = points[:] + points[0:2]  # repeat first two points at end
    angles = []
    for p0, p1, p2 in zip(points, points[1:], points[2:]):
        v0 = vector(tail=p0, head=p1)
        a0 = math.atan2(v0[1], v0[0])
        v1 = vector(tail=p1, head=p2)
        a1 = math.atan2(v1[1], v1[0])
        angle = a1 - a0
        if angle < 0:
            angle += 2 * math.pi
        angles.append(angle)
    return angles


def angles_to_centroid(points, center):
    pts = np.array([np.array(p) for p in points])
    center = np.array(center)
    angles = []
    for p_1 in pts:
        p_2 = np.array(p_1[0], center[1])

        cp_1 = p_1 - center
        cp_2 = p_2 - center

        cosine = np.dot(cp_1, cp_2) / (
                    np.linalg.norm(cp_1) * np.linalg.norm(cp_2))
        angle = np.arccos(cosine)
        angles.append(angle)  # np.degrees(angle)
    return angles


def calculate_edges_lengths(points):
    points = points[:] + points[0:1]  # repeat first point at end
    lengths = []
    for p0, p1 in zip(points, points[1:]):
        lengths.append(math.dist(p0, p1))
    return lengths


def get_most_similar_prototype(prototypes, video, volume_tracings_dict,
                               weights=[0.5, 0.5]):
    # get polygon of current instance
    instance_polygon = Polygon(zip(
        ast.literal_eval(volume_tracings_dict[video.file_name]['X']),
        ast.literal_eval(volume_tracings_dict[video.file_name]['Y'])
    ))
    instance_points = [list(x) for x in zip(
        ast.literal_eval(volume_tracings_dict[video.file_name]['X']),
        ast.literal_eval(volume_tracings_dict[video.file_name]['Y'])
    )]
    # compare given image to all prototypes of corresponding volume cluster
    # using euclidean distance and cosine similarity
    # and return the one with the smallest/minimum distance for both metrics
    euc_feature_diff = []
    cosine_feature_diff = []
    iou = []
    shape_diff = []
    i = 0
    for prototype in prototypes:
        if not len(instance_points) == 40 \
                or not len(prototypes[i].segmentation['X']) == 40:
            print(prototypes[i].segmentation['X'])
            print("skipped")
            continue
        euc_feature_diff.append(np.linalg.norm([np.array(video.features) - np.array(prototype.features)]))
        cosine_feature_diff.append(-1 * (cosine_similarity(video.features, [prototype.features])[0][0]))

        # Intersection of Union (IoU) of left ventricle polygons
        prototype_polygon = Polygon(zip(
            prototypes[i].segmentation['X'],
            prototypes[i].segmentation['Y']
        ))
        intersection_polygon = instance_polygon.intersection(prototype_polygon)
        intersection = intersection_polygon.area
        union = prototype_polygon.area + instance_polygon.area - intersection
        iou.append(-1 * (intersection / union))

        # shape similarity using polygon representations
        # by normalized coordinates, angles and rotations
        shape_diff.append(compare_polygons_rotation_translation_invariant(prototype, instance_points))

        # shape similarity using polygon representations
        # by edge lengths and angles between adjacent edges
        # (to use, uncomment following line and comment out the code line above)
        # shape_diff.append(pq.compare_polygons_with_lengths_and_angles(prototype, instance_points))

        i += 1

    # for standardizing
    transformer = StandardScaler()
    # get index of closest prototype regarding (1) Euclidean distance and
    # regarding (2) cosine similarity for similarity of feature vectors
    euc_index = get_most_similar_prototype_index(euc_feature_diff, shape_diff, transformer, weights)
    cosine_index = get_most_similar_prototype_index(cosine_feature_diff, angle_diff, transformer, weights)
    return prototypes[euc_index], euc_index, euc_feature_diff[euc_index], -1 * iou[euc_index], angle_diff[euc_index], \
           prototypes[cosine_index], cosine_index, -1 * cosine_feature_diff[cosine_index], -1 * iou[cosine_index], angle_diff[cosine_index]


def get_most_similar_prototype_index(feature_diff, shape_diff, transformer,
                                     weights):
    # calculate standardized weighted sum
    evaluation = [list(x) for x in zip(feature_diff, iou, shape_diff)]
    eval_ft = transformer.fit_transform(evaluation)
    weighted_sum = [weights[0] * x[0] + weights[1] * x[1] for x in eval_ft]
    # get index of prototype with minimum difference
    most_similar_index = weighted_sum.index(min(weighted_sum))
    return most_similar_index


def get_most_similar_prototype_ssim(prototypes, image, features=True):
    most_similar_index = 0
    most_similar = prototypes[most_similar_index]
    if features:
        max_sim = structural_similarity(
                np.array(image.features[0]).astype('float64'), np.array(most_similar.features),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    else:
        max_sim = structural_similarity(
                np.array(image.image[0]), np.array(most_similar.image),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                multichannel=True)
    i = 1
    for prototype in prototypes[1:]:
        if features:
            current_sim = structural_similarity(
                np.array(image.features[0]).astype('float64'), np.array(image.features),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        else:
            current_sim = structural_similarity(
                np.array(image.image[0]), np.array(prototype.image),
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                multichannel=True)
        if current_sim > max_sim:
            most_similar = prototype
            most_similar_index = i
            max_sim = current_sim
        i += 1
    return most_similar, most_similar_index


def get_most_similar_prototype_psnr(prototypes, image, features=True):
    most_similar_index = 0
    most_similar = prototypes[most_similar_index]
    if features:
        max_sim = peak_signal_noise_ratio(
            np.array(image.features[0]), np.array(most_similar.features),
            data_range=max(np.array(image.features[0]) - min(np.array(image.features[0])))
        )
    else:
        max_sim = peak_signal_noise_ratio(
            np.array(image.image[0]), np.array(most_similar.image),
            data_range=max(np.array(image.image[0])) - min(np.array(image.image[0]))
        )
    i = 1
    for prototype in prototypes[1:]:
        if features:
            current_sim = peak_signal_noise_ratio(
                np.array(image.features[0]), np.array(prototype.features),
                data_range=max(np.array(image.features[0])) - min(np.array(image.features[0]))
            )
        else:
            current_sim = peak_signal_noise_ratio(
                np.array(image.image[0]), np.array(prototype.image),
                data_range=max(np.array(image.image[0])) - min(np.array(image.image[0]))
            )
        if current_sim > max_sim:
            most_similar = prototype
            most_similar_index = i
            max_sim = current_sim
        i += 1
    return most_similar, most_similar_index
