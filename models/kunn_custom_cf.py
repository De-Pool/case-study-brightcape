import pathlib

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import split_data as split
import test_model


class CollaborativeFilteringKUNNCustom(object):
    def __init__(self, filename, model_data, k_c, k_p, alpha=0, plot=False, save=False):
        self.k_c = k_c
        self.k_p = k_p
        self.alpha = alpha
        self.save = save

        if save:
            # Create the /product_cf directory
            pathlib.Path('../data/kunn_custom').mkdir(parents=True, exist_ok=True)

        if not isinstance(model_data, dict):
            # data can also be: temporal, last_out, one_out
            data = split.create_model_data(filename, plot, model_data)

        self.matrix = model_data['matrix']
        self.train_matrix = model_data['train_matrix']
        self.customers_map = model_data['customers_map']
        self.products_map = model_data['products_map']
        self.test_data = model_data['test_data']
        self.df_clean = model_data['df_clean']
        self.n = model_data['n']
        self.m = model_data['m']
        self.similarity_matrix_p = None
        self.similarity_matrix_c = None
        self.ratings_matrix = None

    def create_similarity_matrices(self):
        # For each customer and for each product, compute how similar they are to each other customer and product.
        self.similarity_matrix_c = cosine_similarity(sparse.csr_matrix(self.train_matrix))
        self.similarity_matrix_p = cosine_similarity(sparse.csr_matrix(self.train_matrix.T))

    def predict_ratings_matrix(self, bought_before=True):
        customers_filter = (np.argsort(np.argsort(self.similarity_matrix_c, axis=1)) >=
                            self.similarity_matrix_c.shape[
                                1] - self.k_c) * 1
        np.fill_diagonal(customers_filter, 0)
        products_filter = (np.argsort(np.argsort(self.similarity_matrix_p, axis=1)) >=
                           self.similarity_matrix_p.shape[
                               1] - self.k_p) * 1
        np.fill_diagonal(products_filter, 0)

        ratings_matrix_c = ((customers_filter @ self.train_matrix) / self.k_c)
        ratings_matrix_p = ((products_filter @ self.train_matrix.T) / self.k_p).T
        self.ratings_matrix = ratings_matrix_c + ratings_matrix_p
        if bought_before:
            # Set each rating to 0 for products which have already been bought.
            nonzero = np.where(self.train_matrix == 1, 0, 1)
            self.ratings_matrix = self.ratings_matrix * nonzero

    def fit(self):
        self.create_similarity_matrices()
        self.predict_ratings_matrix()


def gridsearch(model_data, k_cs, k_ps, similar_items=False, similar_products_dict=None):
    best_kunn_custom = [0]
    best_params_kunn_custom = []
    all_params_kunn_custom = []

    model_kunn_custom = CollaborativeFilteringKUNNCustom('', model_data, 1, 1, 0, False, False)
    model_kunn_custom.create_similarity_matrices()
    for k_c in k_cs:
        for k_p in k_ps:
            model_kunn_custom.k_p = k_p
            model_kunn_custom.k_c = k_c

            model_kunn_custom.predict_ratings_matrix()
            performance_kunn_custom = test_model.all_methods(model_kunn_custom, model_data['r'], similar_items,
                                                             similar_products_dict)

            all_params_kunn_custom.append([k_c, k_p, performance_kunn_custom])
            if performance_kunn_custom[0] > best_kunn_custom[0]:
                best_kunn_custom = performance_kunn_custom
                best_params_kunn_custom = [k_c, k_p]

    return best_kunn_custom, best_params_kunn_custom, all_params_kunn_custom
