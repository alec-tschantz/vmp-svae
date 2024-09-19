import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_clusters(original_data, reconstructed_data, cluster_indices):
    plt.figure(figsize=(6, 4))

    unique_clusters = np.unique(cluster_indices)
    colors = plt.cm.get_cmap("tab10", len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        cluster_mask = cluster_indices == cluster
        plt.scatter(
            reconstructed_data[cluster_mask, 0],
            reconstructed_data[cluster_mask, 1],
            color=colors(i),
            alpha=0.6,
            label=f"Cluster {cluster}",
        )

    plt.scatter(original_data[:, 0], original_data[:, 1], c="grey", alpha=0.3, marker="x", label="Original Data")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()
