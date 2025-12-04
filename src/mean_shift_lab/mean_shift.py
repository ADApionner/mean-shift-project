import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class MyMeanShift(BaseEstimator, ClusterMixin):
    def __init__(self, bandwidth=2.0):
        self.bandwidth = bandwidth  #
        self.centroids_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.array(X)
        points = np.copy(X)  # Kopia robocza punktów

        # 1. Przesuwanie punktów (Hill Climbing)
        for _ in range(100):  # Max 100 iteracji
            new_points = []
            for p in points:
                # Oblicz odległości do wszystkich innych punktów
                dists = np.linalg.norm(X - p, axis=1)
                # Wybierz te w promieniu (Flat Kernel)
                neighbors = X[dists < self.bandwidth]
                # Przesuń punkt do średniej jego sąsiadów
                new_points.append(neighbors.mean(axis=0))

            new_points = np.array(new_points)
            # Jeśli punkty przestały się ruszać -> koniec
            if np.linalg.norm(new_points - points) < 1e-3:
                points = new_points
                break
            points = new_points

        # 2. Grupowanie bliskich centrów (Merging)
        unique_centers = []
        for p in points:
            if not unique_centers:
                unique_centers.append(p)
            else:
                # Sprawdź czy ten punkt jest blisko któregoś z już zapisanych
                dists = np.linalg.norm(unique_centers - p, axis=1)
                if np.min(dists) > self.bandwidth:
                    unique_centers.append(p)

        self.centroids_ = np.array(unique_centers)

        # 3. Przypisanie etykiet (który punkt do którego klastra)
        self.labels_ = []
        for x in X:
            dists = np.linalg.norm(self.centroids_ - x, axis=1)
            self.labels_.append(np.argmin(dists))
        self.labels_ = np.array(self.labels_)

        return self