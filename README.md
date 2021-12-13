# Prototype-based Explanation of Neural Network Outputs for Ultrasound Images

This repository contains a method for calculating prototypes of
echocardiographic still images in order to allow a validation of
neural network outputs by comparing a new input to an already
labeled similar prototypical still image. Here, the prediction
of the blood volume of the left ventricle of the heart is considered, but it can
be any other predicted parameter.

In order to apply the method using a dataset which consists of echocardiograms
with corresponding known ESV/EDV frame positions, volumes, and 
possibly segmentations of the left ventricle in the form of
coordinate pairs, the following steps have to be executed:

## Preprocessing: Extraction of Still Images
Execute *still_images_preprocessing*  to get the still images that
represent the ESV and EDV time points. A corresponding
file *frame_volumes.csv* with the metadata is generated.

## Clustering of predicted Volumes
The file *clustering_volumes* generates the files that contain
the interval borders for the volume ranges as well as the instances
labeled with their assigned interval indices.

## Extraction of latent Features
Before clustering the volume subgroups, the latent features
have to be extracted from a hidden layer of the model to be 
explained by use of *image_feature_extraction*.
This has to be done for each desired hidden layer separately.

## Clustering of Volume Subgroups by using the latent Features
After feature extraction, the volume subgroups are clustered
internally by *clustering_images*, which makes use of **kmedoids**.
The obtained cluster centers can then be saved as prototypes
with *prototypes_calculation*.

## Evaluation of calculated Prototypes 
The file *prototypes_evaluation* applies methods for different
variants of similarity measurement for selecting and evaluating the 
calculated prototypes. The weights parameter in the function call of
*get_most_similar_prototype* (*similarity*) as well as the
distance calculation in the method can be adjusted slightly depending on 
the required variant. Also, dynamic time warping can be used.
For this purpose, to reduce the run time, adjusting the following part of 
*compare_polygons_rotation_translation_invariant* is necessary:
```python
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
```
The standard variant of get_most_similar_prototype* is the weighted sum
combining half of feature distance and half of shape distance
(which is calculated by using Euclidean distance).

## Model
It's possible to use an individual model with input shape (112x112x3)
or the given ResNet-18 with additional dense layers 
(see *two_d_resnet*).

## Visualization and Analysis
Files contained in *results_analysis* are meant for
visualizing or saving e.g. correlations, applicability domains,
ranks or instances/segmentations.

