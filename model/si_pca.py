import numpy as np
from inc_pca import IncPCA


class StreamingIPCA:
    def __init__(self, n_components, forgetting_factor=1.0):
        self.n_components = n_components
        self.pca_model = IncPCA(n_components, forgetting_factor)
        self.total_data = None
        self.pre_embeddings = None

    def fit_new_data(self, x, labels=None):
        if self.total_data is None:
            self.total_data = x
        else:
            self.total_data = np.concatenate([self.total_data, x], axis=0)

        self.pca_model.partial_fit(x)
        cur_embeddings = self.pca_model.transform(self.total_data)

        if self.pre_embeddings is None:
            self.pre_embeddings = cur_embeddings
        else:
            self.pre_embeddings = IncPCA.geom_trans(self.pre_embeddings, cur_embeddings)

        return self.pre_embeddings
