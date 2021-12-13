from sklearn import metrics
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from statsmodels.nonparametric.bandwidths import select_bandwidth
import numpy as np
import matplotlib.pyplot as plt
import jenkspy
import math
import pandas


def jenks_caspall(data, sort_col, n_clusters):
    data.sort_values(by=sort_col)
    breaks = jenkspy.jenks_breaks(data[sort_col], nb_class=n_clusters)
    labels = [str(i) for i in range(0, n_clusters)]
    print('Number of Clusters', len(labels))
    data['Cluster'] = pandas.cut(data[sort_col],
                                 bins=breaks,
                                 labels=labels,
                                 include_lowest=True)
    return breaks, data


def k_medoids_comparing_cluster_number(data, n, max_iter=500, plot=False,
                                       all_scores=False):
    # try each value between 2 and given maximum number of clusters
    kme = [KMedoids(n_clusters=k, init='window_size-medoids++', max_iter=max_iter)
           for k in range(2, n + 1)]
    return compare_n_clusters(kme, data, n, plot, all_scores)


def compare_n_clusters(cl_instances, data, n, plot=False, all_scores=False):
    sse = []
    for k in range(0, n - 1):
        sse.append(cl_instances[k].fit(data).inertia_)
    if all_scores:
        # score: opposite of the distance between the data samples and their
        # associated cluster centers (which is our objective)
        scores = [(cl_instances[k].fit(data).score(
            data)) for k in range(0, n - 1)]
        silhouette_scores = [metrics.silhouette_score(
            data, cl_instances[k].fit(data).labels_) for k in range(0, n - 1)]
        dbs = [metrics.davies_bouldin_score(
            data, cl_instances[k].fit(data).labels_) for k in range(0, n - 1)]
        chs = [metrics.calinski_harabasz_score(
            data, cl_instances[k].fit(data).labels_) for k in range(0, n - 1)]
    if plot:
        plot_scores(range(2, n + 1), sse, 'SSE')
        if all_scores:
            plot_scores(range(2, n + 1), scores, 'K-Means Scores')
            plot_scores(range(2, n + 1), silhouette_scores, 'Silhouetten coefficient')
            plot_scores(range(2, n + 1), dbs, 'db')
            plot_scores(range(2, n + 1), chs, 'ch')
        plt.show()
    best_index = find_best_cluster_number(sse)
    print('Best found number of clusters regarding SSE: ' + str(best_index + 2))
    cl_instances[best_index].fit_transform(data)
    cluster_labels = cl_instances[best_index].predict(data)
    cluster_centers = cl_instances[best_index].cluster_centers_
    if all_scores:
        return cluster_labels, cluster_centers, sse, silhouette_scores, dbs, chs
    return cluster_labels, cluster_centers, sse


# using elbow method: find elbow of the curve corresponding to SSE
# calculate for each possible number of clusters the distance from the
# corresponding point of the curve to the straight line defined by
# the first and the last point of the curve
def find_best_cluster_number(measures):
    largest_distance = 0.0
    best_number = 0
    if len(measures) <= 1:
        return best_number
    for i in range(0, len(measures)):
        distance = distance_point_to_line(
            i + 2, abs(measures[i]),
            2, abs(measures[0]),
            len(measures) + 1, abs(measures[len(measures) - 1])
        )
        if distance > largest_distance:
            largest_distance = distance
            best_number = i
        else:
            break
    return best_number


def distance_point_to_line(x0, y0, x1, y1, x2, y2):
    # Calculates distance from point (x0, y0) to line defined by
    # (x1, y1) and (x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return float('inf')
    distance = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx * dx + dy * dy)
    return distance


def plot_scores(nc, scores, title):
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.scatter(nc, scores)
    plt.xlabel('Number of clusters K', fontsize=17)
    plt.ylabel(title, fontsize=17)
    plt.show()


def kde(data, bw_method):
    data = np.array(data)
    bandwidth = select_bandwidth(data, bw=bw_method, kernel=None)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
    s = np.linspace(0, 100)
    e = kde.score_samples(s.reshape(-1, 1))
    # minima correspond to interval borders, maxima to centroids
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    return s[mi], s[ma]
