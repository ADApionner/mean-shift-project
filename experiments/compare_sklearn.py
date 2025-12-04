import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from mean_shift_lab.mean_shift import MyMeanShift


def run():
    # --- KONFIGURACJA DANYCH ---
    # 1. Definiujemy środki grup, żeby były blisko siebie (trójkąt)
    centers = [[1, 1], [3, 3], [5, 1]]

    # 2. Generujemy dane:
    # n_samples=1000 -> dużo punktów
    # cluster_std=1.0 -> szerokie grupy
    X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=1.0)

    BANDWIDTH =1.5

    print(f"Uruchamianie MyMeanShift (bandwidth={BANDWIDTH})...")
    my = MyMeanShift(bandwidth=BANDWIDTH)
    my.fit(X)

    print("Uruchamianie Sklearn...")
    sk = MeanShift(bandwidth=BANDWIDTH, bin_seeding=False)
    sk.fit(X)

    # --- RYSOWANIE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Wykres 1: NASZ
    ax1.scatter(X[:, 0], X[:, 1], c=my.labels_, s=15, alpha=0.6)
    ax1.scatter(my.centroids_[:, 0], my.centroids_[:, 1], marker='x', color='red', s=200, linewidths=3,
                label='Centroidy')
    ax1.set_title(f"Mój algorytm: {len(my.centroids_)} klastry")
    ax1.legend()

    # Wykres 2: Sklearn
    ax2.scatter(X[:, 0], X[:, 1], c=sk.labels_, s=15, alpha=0.6)
    ax2.scatter(sk.cluster_centers_[:, 0], sk.cluster_centers_[:, 1], marker='x', color='red', s=200, linewidths=3,
                label='Centroidy')
    ax2.set_title(f"Sklearn: {len(sk.cluster_centers_)} klastry")
    ax2.legend()

    plt.suptitle("Porównanie na gęstych, bliskich danych", fontsize=16)
    plt.show()


if __name__ == "__main__":
    run()