import numpy as np
from mean_shift_lab.mean_shift import MyMeanShift


def test_algorithm():
    # Dane: dwie grupy punktów daleko od siebie
    X = np.array([[1, 1], [1.1, 1.1], [10, 10], [10.1, 10.1]])

    ms = MyMeanShift(bandwidth=2.0)
    ms.fit(X)

    # Musi znaleźć 2 klastry
    assert len(ms.centroids_) == 2
    # Etykiety muszą dzielić zbiór 2 na 2 (np. [0,0, 1,1])
    assert len(np.unique(ms.labels_)) == 2


def test_single_cluster():
    """Wszystkie punkty bardzo blisko siebie powinny dać 1 klaster."""
    X = np.random.rand(10, 2) * 0.1  # Punkty w kwadracie 0.1 x 0.1

    ms = MyMeanShift(bandwidth=1.0)  # Bandwidth 1.0 jest znacznie większe niż rozrzut danych
    ms.fit(X)

    assert len(ms.centroids_) == 1


def test_fit_returns_self():
    """Zgodność z API sklearn - metoda fit musi zwracać self."""
    X = np.array([[1, 1], [2, 2]])
    ms = MyMeanShift()
    result = ms.fit(X)
    assert result is ms