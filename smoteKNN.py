import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


class SMOTEKNN(SMOTE):
    def __init__(self, k_neighbors=5, **kwargs):
        super(SMOTEKNN, self).__init__(**kwargs)
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        X_resampled, y_resampled = super(SMOTEKNN, self).fit_resample(X, y)

        knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        knn.fit(X_resampled, y_resampled)

        majority_class = Counter(y.ravel().tolist()).most_common(1)[0][0]
        delete_indices = []

        for i, x in enumerate(X_resampled[len(X):]):
            if i >= len(X) and knn.predict([x])[0] == majority_class:
                delete_indices.append(i)

        X_resampled_filtered = np.delete(X_resampled, delete_indices, axis=0)
        y_resampled_filtered = np.delete(y_resampled, delete_indices)

        return X_resampled_filtered, y_resampled_filtered
