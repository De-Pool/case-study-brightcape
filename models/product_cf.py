import pathlib

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import split_data as split
import test_model


class CollaborativeFilteringProduct(object):
    def __init__(self, filename, data, k, alpha=0, plot=False, save=False):
        self.k = k
        self.alpha = alpha
        self.save = save

        if save:
            # Create the /product_cf directory
            pathlib.Path('../data/product_cf').mkdir(parents=True, exist_ok=True)

        if not isinstance(data, dict):
            # data can also be: temporal, last_out, one_out
            data = split.create_model_data(filename, plot, data)

        self.matrix = data['matrix'].T
        self.train_matrix = data['train_matrix'].T
        self.customers_map = data['customers_map']
        self.products_map = data['products_map']
        self.test_data = data['test_data']
        self.df_clean = data['df_clean']
        self.n = data['n']
        self.m = data['m']

        self.similarity_matrix = None
        self.ratings_matrix = None

    def create_similarity_matrix(self):
        # For each customer, compute how similar they are to each other customer.
        if config.use_cupy:
            self.similarity_matrix = np.asarray(cosine_similarity(sparse.csr_matrix(np.asnumpy(self.train_matrix))))
        else:
            self.similarity_matrix = cosine_similarity(sparse.csr_matrix(self.train_matrix))

    def predict_ratings_matrix(self, bought_before=True):
        products_filter = (np.argsort(np.argsort(self.similarity_matrix, axis=1)) >=
                           self.similarity_matrix.shape[
                               1] - self.k) * 1
        np.fill_diagonal(products_filter, 0)

        self.ratings_matrix = ((products_filter @ self.train_matrix) / self.k).T

        if bought_before:
            # Set each rating to 0 for products which have already been bought.
            nonzero = np.where(self.train_matrix.T == 1, 0, 1)
            self.ratings_matrix = self.ratings_matrix * nonzero

    def fit(self):
        self.create_similarity_matrix()
        self.predict_ratings_matrix()


def gridsearch(data, k_s, similar_items=False, similar_products_dict=None):
    best_basic = [0]
    best_param_basic = 0
    all_params_basic = []
    model_product_cf = CollaborativeFilteringProduct('', data, 1, 0, False, False)
    model_product_cf.fit()
    for k in k_s:
        model_product_cf.k = k
        model_product_cf.predict_ratings_matrix()
        performance_basic = test_model.all_methods(model_product_cf, data['r'], similar_items, similar_products_dict)
        all_params_basic.append([k, performance_basic])
        if performance_basic[0] > best_basic[0]:
            best_basic = performance_basic
            best_param_basic = k
    return best_basic, best_param_basic, all_params_basic
