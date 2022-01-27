import pathlib

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import split_data as split


class CollaborativeFilteringKUNNCustom(object):
    def __init__(self, filename, data, k_c, k_p, alpha=0, plot=False, save=False):
        self.k_c = k_c
        self.k_p = k_p
        self.alpha = alpha
        self.save = save

        if save:
            # Create the /product_cf directory
            pathlib.Path('../data/product_cf').mkdir(parents=True, exist_ok=True)

        if not isinstance(data, dict):
            # data can also be: temporal, last_out, one_out
            data = split.create_model_data(filename, plot, data)

        self.matrix = data['matrix']
        self.train_matrix = data['train_matrix']
        self.customers_map = data['customers_map']
        self.products_map = data['products_map']
        self.test_data = data['test_data']
        self.df_clean = data['df_clean']
        self.n = data['n']
        self.m = data['m']
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
