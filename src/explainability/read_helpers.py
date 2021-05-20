
class Video:
    def __init__(self, features, ef, file_name, video=None):
        self.features = features
        self.ef = ef
        self.file_name = file_name
        self.video = video


def read_cluster_labels(cluster_file):
    cluster_labels = []
    efs = []
    file_names = []
    with open(cluster_file, "r") as txt_file:
        for line in txt_file:
            line_split = line.split(" ")
            cluster_labels.append(int(line_split[0]))

            efs.append(float(line_split[1]))
            file_names.append(line_split[2].rsplit()[0])
    return cluster_labels, efs, file_names


def read_extracted_features(file_path):
    extracted_features = []
    with open(file_path, "r") as txt_file:
        for line in txt_file:
            if not line.startswith("tf.Tensor"):
                if line.startswith("[["):
                    video_features = []
                    line = line.strip("[")
                if line.__contains__("shape"):
                    line = line.split("]")[0]
                    video_features.extend([float(v) for v in line.split()])
                    extracted_features.append(video_features)
                else:
                    video_features.extend([float(v) for v in line.split()])
    return extracted_features


def read_video_clusters(cluster_labels_file, video_features_file):
    # list of clusters
    # where each cluster is a list of its corresponding videos
    cluster_features = []

    video_cluster_labels, ef_file, file_names = \
        read_cluster_labels(cluster_labels_file)
    num_clusters = max(video_cluster_labels) + 1
    print(str(cluster_labels_file), " num clusters: ", num_clusters)
    for i in range(num_clusters):
        cluster_features.append([])

    video_features = read_extracted_features(video_features_file)
    for i in range(len(video_features)):
        cluster_features[video_cluster_labels[i]].append(
            Video(video_features[i], ef_file[i], file_names[i]))
    return cluster_features


def read_video_cluster_centers(centers_file_path, video_known=True):
    cluster_centers = []
    new_center = True
    if video_known:
        i = 0
    else:
        i = -2
    with open(centers_file_path, "r") as txt_file:
        for line in txt_file:
            line_split = line.split()
            if new_center:
                new_center = False
                if video_known:
                    ef = line_split[1]
                    file_name = line_split[2]
                else:
                    ef = None
                    file_name = None
                if len(line_split[3 + i].strip("[")) == 0:
                    features = []
                else:
                    features = [float(line_split[3 + i].strip("["))]
                for f in line_split[4 + i:]:
                    features.append(float(f))
            else:
                end = len(line_split) - 1
                if line_split[end].endswith("]"):
                    line_split[end] = line_split[end].strip("]")
                    if len(line_split[end]) == 0:
                        line_split.pop()
                    for f in line_split:
                        features.append(float(f))
                    cluster_centers.append(Video(features, ef, file_name))
                    new_center = True
                else:
                    for f in line_split:
                        features.append(float(f))
    return cluster_centers


def read_ef_cluster_centers(centers_file_path):
    cluster_centers = []
    with open(centers_file_path, "r") as txt_file:
        for line in txt_file:
            line_split = line.split()
            ef = line_split[1].strip("[]")
            cluster_centers.append(float(ef))
    return cluster_centers


def read_prototypes(centers_file_path):
    prototypes = {}
    new_center = True
    ef_cluster_index = 0
    prototypes[ef_cluster_index] = []
    with open(centers_file_path, "r") as txt_file:
        for line in txt_file:
            line_split = line.split()
            if new_center:
                new_center = False
                if int(line_split[0]) is not ef_cluster_index:
                    ef_cluster_index = int(line_split[0])
                    prototypes[ef_cluster_index] = []
                if not line_split[2] == 'None':
                    ef = float(line_split[2])
                else:
                    ef = 0
                if not line_split[3] == 'None':
                    file_name = line_split[3]
                else:
                    file_name = None
                if len(line_split[4].strip("[")) == 0:
                    features = []
                else:
                    features = [float(line_split[4].strip("["))]
                for f in line_split[5:]:
                    features.append(float(f))
            else:
                end = len(line_split) - 1
                if line_split[end].endswith("]"):
                    line_split[end] = line_split[end].strip("]")
                    if len(line_split[end]) == 0:
                        line_split.pop()
                    for f in line_split:
                        features.append(float(f))
                    prototypes[ef_cluster_index].append(Video(features, ef, file_name))
                    new_center = True
                else:
                    for f in line_split:
                        features.append(float(f))
    return prototypes

