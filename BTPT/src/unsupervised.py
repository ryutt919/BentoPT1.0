import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import argparse


def load_features(data_dir):
    features = []
    file_paths = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for name in os.listdir(class_dir):
            if not name.endswith('.npz'):
                continue
            path = os.path.join(class_dir, name)
            data = np.load(path, allow_pickle=True)
            kpts = data['keypoints']  # shape (T, 13, 3)
            # frame 평균을 사용하고 confidence는 무시
            mean_xy = kpts[:, :, :2].mean(axis=0).flatten()
            features.append(mean_xy)
            file_paths.append(path)
    return np.array(features), file_paths


def main():
    parser = argparse.ArgumentParser(description='Cluster normalized keypoints')
    parser.add_argument('--data_dir', default='mmaction2/data/kinetics400/normalized',
                        help='Directory with normalized keypoints')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive scatter plot')
    args = parser.parse_args()

    X, paths = load_features(args.data_dir)
    if len(X) == 0:
        print('No data found')
        return

    kmeans = KMeans(n_clusters=args.clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    for p, label in zip(paths, labels):
        print(f'{p}: cluster {label}')

    if args.interactive:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        fig = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=labels.astype(str),
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'cluster'},
            title='KMeans Clusters')
        fig.show()


if __name__ == '__main__':
    main()