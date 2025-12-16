import numpy as np
from mean_shift_lab.mean_shift import MyMeanShift


def test_algorithm():
    # Data: two groups of points far apart
    X = np.array([[1, 1], [1.1, 1.1], [10, 10], [10.1, 10.1]])

    ms = MyMeanShift(bandwidth=2.0)
    ms.fit(X)

    # Must find 2 clusters
    assert len(ms.centroids_) == 2
    # Labels must split the set 2 by 2 (e.g., [0,0, 1,1])
    assert len(np.unique(ms.labels_)) == 2


def test_single_cluster():
    """All points very close to each other should result in 1 cluster."""
    X = np.random.rand(10, 2) * 0.1  # Points in a 0.1 x 0.1 square

    ms = MyMeanShift(bandwidth=1.0)  # Bandwidth 1.0 is much larger than data spread
    ms.fit(X)

    assert len(ms.centroids_) == 1

