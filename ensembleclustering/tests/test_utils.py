# test_utils.py

import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from ..ClusterEnsembles import cluster_ensembles


def test_all():
    base_clusters = np.array([
        [1, 1, 1, 2, 2, 3, 3],
        [2, 2, 2, 3, 3, 1, 1],
        [4, 4, 2, 2, 3, 3, 3],
        [1, 2, np.nan, 1, 2, np.nan, np.nan]
    ])
    label_true = np.array([1, 1, 1, 2, 2, 3, 3])
    label_pred = cluster_ensembles(base_clusters, solver='all', random_state=0)
    nmi_score = normalized_mutual_info_score(label_true, label_pred, average_method='geometric')
    assert nmi_score == 1.0

