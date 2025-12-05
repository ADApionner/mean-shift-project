import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class MyMeanShift(BaseEstimator, ClusterMixin):
    def __init__(self, bandwidth=2.0):
        self.bandwidth = bandwidth
        self.centroids_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.array(X)
        points = np.copy(X)  # Working copy of points

        # 1. Shifting points (Hill Climbing)
        for _ in range(100):  # Max 100 iterations
            new_points = []
            for p in points:
                # Calculate distances to all other points
                dists = np.linalg.norm(X - p, axis=1)
                # Select points within radius (Flat Kernel)
                neighbors = X[dists < self.bandwidth]
                # Shift point to the mean of its neighbors
                new_points.append(neighbors.mean(axis=0))

            new_points = np.array(new_points)
            # If points stopped moving -> stop
            if np.linalg.norm(new_points - points) < 1e-3:
                points = new_points
                break
            points = new_points

        # 2. Grouping close centers (Merging)
        unique_centers = []
        for p in points:
            if not unique_centers:
                unique_centers.append(p)
            else:
                # Check if this point is close to any already saved center
                dists = np.linalg.norm(unique_centers - p, axis=1)
                if np.min(dists) > self.bandwidth:
                    unique_centers.append(p)

        self.centroids_ = np.array(unique_centers)

        # 3. Assign labels (which point belongs to which cluster)
        self.labels_ = []
        for x in X:
            dists = np.linalg.norm(self.centroids_ - x, axis=1)
            self.labels_.append(np.argmin(dists))
        self.labels_ = np.array(self.labels_)

        return self