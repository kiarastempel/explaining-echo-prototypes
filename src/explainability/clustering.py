import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from statsmodels.nonparametric.bandwidths import select_bandwidth
import jenkspy
import Plot.clusters as pcl
import math
import pandas


def k_means(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    centroids = kmeans.cluster_centers_

    plt.scatter(data, np.zeros_like(data),
                c=kmeans.labels_.astype(float), s=0.5,
                alpha=0.5)
    plt.scatter(centroids, np.zeros_like(centroids), c='red', s=10)
    plt.show()
    return centroids


def kde(data, bw_method):
    data = np.array(data)
    bandwidth = select_bandwidth(data, bw=bw_method, kernel=None)
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(data)
    s = np.linspace(0, 100)
    e = kde.score_samples(s.reshape(-1, 1))

    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    print("Minima / interval border:", s[mi])
    print("Maxima / centroids:", s[ma])

    plt.plot(s, e, 'b',
             s[ma], e[ma], 'go',
             s[mi], e[mi], 'ro',
             linewidth=1)
    plt.xlabel("EF")
    plt.ylabel("Density")
    plt.title("Bandwidth = " + str(round(bandwidth[0], 4)) + " (method: " + bw_method + ")")
    plt.show()
    return s[mi], s[ma]


def jenks_caspall(data, sort_col, n_clusters):
    data.sort_values(by=sort_col)
    breaks = jenkspy.jenks_breaks(data['EF'], nb_class=n_clusters)
    labels = [str(i) for i in range(0, n_clusters)]
    print("Length labels", len(labels))
    print("Length bins", len(breaks))
    data['Cluster'] = pandas.cut(data['EF'],
                                 bins=breaks,
                                 labels=labels,
                                 include_lowest=True)
    # print(data[['FileName', 'Cluster']])
    return breaks, data


def find_n_clusters(measures):
    largest_distance = 0.0
    best_index = 0
    if len(measures) <= 1:
        return best_index
    for i in range(0, len(measures)):
        distance = distance_to_line(i + 2, abs(measures[i]),
                                    2, abs(measures[0]),
                                    len(measures) + 1, abs(measures[len(measures) - 1]))
        if distance > largest_distance:
            largest_distance = distance
            best_index = i
        else:
            break
    return best_index


def distance_to_line(x0, y0, x1, y1, x2, y2):
    # Calculates distance from (x0,y0) to line defined by (x1,y1) and (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    # if dx == 0 and dy == 0:
    #     return float('inf')
    return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx * dx + dy * dy)


def compare_n_clusters_k_medoids(data, n, max_iter=500, plot=True,
                                 all_scores=False, plot_name=""):
    # try each possible value for n_clusters
    kme = [KMedoids(n_clusters=k, init='k-medoids++', max_iter=max_iter)
           for k in range(2, n + 1)]
    print("n", n)
    return compare_n_clusters(kme, data, n, plot, all_scores, plot_name)


def compare_n_clusters_k_means(data, n, runs=30, max_iter=500, tol=0.0000000001,
                               plot=True, all_scores=False, plot_name=""):
    # try each possible value for n_clusters between 2 and n
    # minimize chance to get a local optimum by running k-means 'runs' times
    km = [KMeans(n_clusters=k, n_init=runs, max_iter=max_iter, tol=tol)
          for k in range(2, n + 1)]
    # show inertia in each iteration using verbose
    # km = [KMeans(n_clusters=k, n_init=runs, max_iter=max_iter, tol=tol,
    #                  verbose=3) for k in range(2, n + 1)]
    return compare_n_clusters(km, data, n, plot, all_scores, plot_name)


def compare_n_clusters(cl_instances, data, n, plot=True, all_scores=False,
                       plot_name=""):
    sse = []
    print("sse calculation started")
    for k in range(0, n - 1):
        sse.append(cl_instances[k].fit(data).inertia_)
        print("Clustering using k =", k, "finished")
    # sse = [cl_instances[k].fit(data).inertia_ for k in range(0, n - 2)]
    print("sse calculation finished")
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
        print("chs calculation finished")
    if plot:
        pcl.plot_silhouette_scores(range(2, n + 1), sse, "Abweichungsquadratsumme")
        if all_scores:
            pcl.plot_silhouette_scores(range(2, n + 1), scores, "K-Means Scores")
            pcl.plot_silhouette_scores(range(2, n + 1), silhouette_scores,
                                       "Silhouetten-Koeffizient")
            pcl.plot_silhouette_scores(range(2, n + 1), dbs, "db")
            pcl.plot_silhouette_scores(range(2, n + 1), chs, "ch")
        plt.show()
        # plt.savefig(plot_name)
        # plt.close()
    best_index = find_n_clusters(sse)
    print("n_clusters: " + str(best_index + 2))
    print("length", len(cl_instances))
    cl_instances[best_index].fit_transform(data)
    cluster_labels = cl_instances[best_index].predict(data)
    cluster_centers = cl_instances[best_index].cluster_centers_
    if all_scores:
        return cluster_labels, cluster_centers, sse, silhouette_scores, dbs, chs
    return cluster_labels, cluster_centers, sse
