import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from mean_shift_lab.mean_shift import MyMeanShift


def run():
    # --- DATA CONFIGURATION ---
    # 1. Define cluster centers to be close to each other (triangle)
    centers = [[1, 1], [3, 3], [5, 1]]

    # 2. Generate data:
    # n_samples=1000 -> many points
    # cluster_std=1.0 -> wide/spread out groups
    X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=1.0)

    BANDWIDTH = 1.5

    print(f"Running MyMeanShift (bandwidth={BANDWIDTH})...")
    my = MyMeanShift(bandwidth=BANDWIDTH)
    my.fit(X)

    print("Running Sklearn...")
    sk = MeanShift(bandwidth=BANDWIDTH, bin_seeding=False)
    sk.fit(X)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: My Implementation
    ax1.scatter(X[:, 0], X[:, 1], c=my.labels_, s=15, alpha=0.6)
    ax1.scatter(my.centroids_[:, 0], my.centroids_[:, 1], marker='x', color='red', s=200, linewidths=3,
                label='Centroids')
    ax1.set_title(f"My Algorithm: {len(my.centroids_)} clusters")
    ax1.legend()

    # Plot 2: Sklearn
    ax2.scatter(X[:, 0], X[:, 1], c=sk.labels_, s=15, alpha=0.6)
    ax2.scatter(sk.cluster_centers_[:, 0], sk.cluster_centers_[:, 1], marker='x', color='red', s=200, linewidths=3,
                label='Centroids')
    ax2.set_title(f"Sklearn: {len(sk.cluster_centers_)} clusters")
    ax2.legend()

    plt.suptitle("Comparison on dense, close data", fontsize=16)
    plt.show()


if __name__ == "__main__":
    run()