import argparse
from pathlib import Path
import pandas
import explainability.clustering as cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', default='../../data',
                        help="Directory with the echocardiograms.")
    parser.add_argument('-o', '--output_directory',
                        help="Directory to save the cluster labels in.")
    parser.add_argument('-m', '--metadata_filename', default='FileList.csv',
                        help="Name of the metadata file.")
    parser.add_argument('--needed_frames', default=50, type=int,
                        help="Number of minimum frames required for an echo "
                             "to be considered.")
    parser.add_argument('-n', '--max_n_clusters', default=100, type=int,
                        help="Maximum number of clusters to be evaluated.")
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    if output_directory is None:
        output_directory = Path(args.input_directory, 'clustering_ef')
    output_directory.mkdir(parents=True, exist_ok=True)

    # get EF of all videos
    metadata_path = Path(args.input_directory, args.metadata_filename)
    file_list_data_frame = pandas.read_csv(metadata_path)
    ef = file_list_data_frame[file_list_data_frame.Split == 'TRAIN'][['FileName', 'EF', 'ESV', 'EDV']]
    ef = ef.append(file_list_data_frame[file_list_data_frame.Split == 'VAL'][['FileName', 'EF', 'ESV', 'EDV']])
    ef = ef.reset_index()

    cluster_by_ef(ef, args.max_n_clusters, output_directory)


def cluster_by_ef(ef, max_n_clusters, output_directory):
    # kmeans
    # cluster_labels, cluster_centers = cl.compare_n_clusters_k_means(
    #     ef[['EF']], max_n_clusters, plot=False)[0:2]
    # cluster_labels, cluster_centers = cl.compare_n_clusters_k_means(
    #     ef[['ESV', 'EDV']], max_n_clusters, plot=True)[0:2]
    # print("kmeans", cluster_labels)
    #
    # with open(Path(output_directory, 'cluster_labels_ef.txt'), "w") as txt_file:
    #     for i in range(len(cluster_labels)):
    #         txt_file.write(str(cluster_labels[i]) + " " + str(ef.at[i, 'EF'])
    #                        + " " + str(ef.at[i, 'FileName']) + "\n")
    # with open(Path(output_directory, 'cluster_centers_ef.txt'), "w") as txt_file:
    #     for i in range(len(cluster_centers)):
    #         txt_file.write(str(i) + " " + str(cluster_centers[i]) + "\n")

    # kmedoids
    # cluster_labels, cluster_centers = cl.compare_n_clusters_k_medoids(
    #     ef[['EF']], max_n_clusters, plot=True)[0:2]
    # print("kmedoids", cluster_labels)
    #
    # with open(Path(output_directory, 'cluster_labels_ef.txt'), "w") as txt_file:
    #     for i in range(len(cluster_labels)):
    #         txt_file.write(str(cluster_labels[i]) + " " + str(ef.at[i, 'EF'])
    #                        + " " + str(ef.at[i, 'FileName']) + "\n")
    # with open(Path(output_directory, 'cluster_centers_ef.txt'), "w") as txt_file:
    #     for i in range(len(cluster_centers)):
    #         txt_file.write(str(i) + " " + str(cluster_centers[i]) + "\n")

    # kernel density estimation
    borders, cluster_centers = cl.kde(ef[['EF']], 'silverman')
    # borders, cluster_centers = cl.kde(ef[['EF']], 'scott')
    # borders, cluster_centers = cl.kde(ef[['EF']], 'normal_reference')

    cluster_labels = []
    for i in range(len(ef[['EF']])):
        current_ef = ef.at[i, 'EF']
        clustered = False
        for j in range(len(borders)):
            if current_ef <= borders[j]:
                cluster_labels.append(j)
                clustered = True
                break
        if not clustered:
            cluster_labels.append(len(borders))

    with open(Path(output_directory, 'cluster_labels_ef.txt'), "w") as txt_file:
        for i in range(len(cluster_labels)):
            txt_file.write(str(cluster_labels[i]) + " " + str(ef.at[i, 'EF'])
                           + " " + str(ef.at[i, 'FileName']) + "\n")

    with open(Path(output_directory, 'cluster_centers_ef.txt'), "w") as txt_file:
        for i in range(len(cluster_centers)):
            txt_file.write(str(i) + " " + str([cluster_centers[i]]) + "\n")

    with open(Path(output_directory, 'cluster_upper_borders_ef.txt'), "w") as txt_file:
        for i in range(len(borders)):
            txt_file.write(str(i) + " " + str([borders[i]]) + "\n")

    # jenks caspall algorithm
    # natural_breaks = cl.jenks_caspall(ef, 'EF', n_clusters=5)
    # print(natural_breaks)


def get_ef_cluster_centers_indices(cluster_centers, ef_list):
    indices = []
    for c in cluster_centers:
        # find echocardiogram that corresponds exactly to ef of center
        for i, ef in enumerate(ef_list):
            if ef == c:
                indices.append(i)
    return indices


if __name__ == '__main__':
    main()
