import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class MyMeanShift(BaseEstimator, ClusterMixin):
    """
    Własna implementacja algorytmu Mean-Shift.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Promień okna (okręgu), w którym szukamy sąsiadów do obliczenia średniej.
        Kluczowy parametr decydujący o liczbie klastrów.

    max_iter : int, default=300
        Maksymalna liczba iteracji przesuwania centroidów.

    tol : float, default=1e-3
        Tolerancja konwergencji. Jeśli centroid przesunie się o mniej niż ta wartość,
        uznajemy, że znalazł "szczyt".
    """

    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.centroids_ = None
        self.labels_ = None

    def fit(self, X):
        """
        Dopasowuje algorytm do danych X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dane treningowe (punkty).

        Returns
        -------
        self : object
            Zwraca instancję klasy.
        """
        # Konwersja do numpy array
        X = np.array(X)

        # Na początku każdy punkt jest potencjalnym centroidem
        # Kopiujemy dane, żeby nie modyfikować oryginału
        current_centroids = np.copy(X)

        for i in range(self.max_iter):
            new_centroids = []
            max_shift = 0.0

            # Dla każdego centroidu szukamy jego nowej pozycji
            for centroid in current_centroids:
                # 1. Oblicz odległość euklidesową od tego centroidu do wszystkich punktów X
                # (Wykorzystujemy broadcasting NumPy dla szybkości)
                distances = np.linalg.norm(X - centroid, axis=1)

                # 2. Znajdź punkty, które są wewnątrz promienia (bandwidth)
                points_within_bandwidth = X[distances < self.bandwidth]

                # 3. Oblicz nową pozycję (średnią z punktów w oknie)
                if len(points_within_bandwidth) > 0:
                    new_center = np.mean(points_within_bandwidth, axis=0)
                else:
                    new_center = centroid  # Jeśli brak sąsiadów, zostajemy w miejscu

                new_centroids.append(new_center)

                # Sprawdzamy o ile przesunął się centroid
                shift = np.linalg.norm(new_center - centroid)
                if shift > max_shift:
                    max_shift = shift

            current_centroids = np.array(new_centroids)

            # Warunek stopu: jeśli żaden centroid nie przesunął się znacząco
            if max_shift < self.tol:
                break

        # --- Post-processing (Grupowanie bliskich centroidów) ---
        # Po konwergencji wiele punktów wyląduje w tym samym miejscu.
        # Musimy usunąć duplikaty (z pewną tolerancją).
        unique_centroids = []
        for centroid in current_centroids:
            if not unique_centroids:
                unique_centroids.append(centroid)
            else:
                # Sprawdź czy ten centroid jest blisko któregokolwiek już zapisanego
                distances = np.linalg.norm(np.array(unique_centroids) - centroid, axis=1)
                if not np.any(distances < self.bandwidth / 2):  # Heurystyka: połowa bandwidth
                    unique_centroids.append(centroid)

        self.centroids_ = np.array(unique_centroids)

        # Przypisanie etykiet (labels) dla każdego punktu X do najbliższego finalnego centroidu
        self.labels_ = self._assign_labels(X, self.centroids_)

        return self

    def _assign_labels(self, X, centroids):
        """Metoda pomocnicza do przypisywania punktów do klastrów."""
        labels = []
        for point in X:
            distances = np.linalg.norm(centroids - point, axis=1)
            labels.append(np.argmin(distances))
        return np.array(labels)