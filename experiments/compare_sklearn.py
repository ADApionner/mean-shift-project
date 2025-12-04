import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift as SklearnMeanShift

from mean_shift_lab.mean_shift import MyMeanShift
# Importujemy naszą klasę (dostosuj ścieżkę importu do swojej struktury folderów)
# from twoja_nazwa_projektu.mean_shift import MyMeanShift
# UWAGA: Aby powyższy import zadziałał, musisz mieć zainstalowany pakiet (pip install -e .)
# lub tymczasowo wkleić klasę MyMeanShift do tego pliku.

def run_experiment():
    # 1. Generowanie sztucznych danych (3 klastry)
    print("Generowanie danych...")
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

    # Parametr bandwidth (promień) - musi być taki sam dla obu algorytmów
    BANDWIDTH = 2.0

    # 2. Uruchomienie Naszej Implementacji
    print("Uruchamianie MyMeanShift...")
    my_ms = MyMeanShift(bandwidth=BANDWIDTH)
    my_ms.fit(X)

    # 3. Uruchomienie Implementacji Scikit-Learn
    print("Uruchamianie Sklearn MeanShift...")
    # Sklearn wymaga jawnego estymowania bandwidth lub podania go. Podajemy, żeby porównanie było uczciwe.
    sklearn_ms = SklearnMeanShift(bandwidth=BANDWIDTH, bin_seeding=True)
    sklearn_ms.fit(X)

    # 4. Wizualizacja Wyników
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Wykres Nasz
    ax1.scatter(X[:, 0], X[:, 1], c=my_ms.labels_, cmap='viridis', marker='o')
    ax1.scatter(my_ms.centroids_[:, 0], my_ms.centroids_[:, 1], c='red', marker='x', s=200, label='Centroidy')
    ax1.set_title(f'MyMeanShift\nZnaleziono klastrów: {len(my_ms.centroids_)}')
    ax1.legend()
    ax1.grid(True)

    # Wykres Scikit-Learn
    ax2.scatter(X[:, 0], X[:, 1], c=sklearn_ms.labels_, cmap='viridis', marker='o')
    ax2.scatter(sklearn_ms.cluster_centers_[:, 0], sklearn_ms.cluster_centers_[:, 1], c='red', marker='x', s=200,
                label='Centroidy')
    ax2.set_title(f'Sklearn MeanShift\nZnaleziono klastrów: {len(sklearn_ms.cluster_centers_)}')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Porównanie implementacji Mean-Shift")
    plt.show()

    # 5. Raport tekstowy
    print("-" * 30)
    print(f"My Implementation Centers: {len(my_ms.centroids_)}")
    print(f"Sklearn Implementation Centers: {len(sklearn_ms.cluster_centers_)}")
    print("-" * 30)


if __name__ == "__main__":
    # Zakładając, że klasa MyMeanShift jest dostępna
    run_experiment()