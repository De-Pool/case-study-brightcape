import pathlib

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import helper_functions as hf
import pre_process_data as ppd
import similarity_meta_data as smd
import split_data as split


class CollaborativeFilteringBasic(object):
    def __init__(self, filename, test_train, k, alpha, plot, save):
        self.k = k
        self.alpha = alpha
        self.save = save

        if save:
            # Create the /basic_cf directory
            pathlib.Path('../data/basic_cf').mkdir(parents=True, exist_ok=True)

        if isinstance(test_train, dict):
            self.matrix = test_train['matrix']
            self.customers_map = test_train['customers_map']
            self.products_map = test_train['products_map']
            self.train_matrix = test_train['train_matrix']
            self.test_data = test_train['test_data']
            self.df_clean = test_train['df_clean']
            self.n = len(self.customers_map)
            self.m = len(self.products_map)
        else:
            self.create_test_train(test_train, filename, plot)

        # Create customer - customer meta data similarity matrix (n x n)
        self.smd_matrix = smd.meta_data_similarity_matrix(self.df_clean, self.customers_map, self.n)

        self.similarity_matrix = None
        self.ratings_matrix = None

    def create_test_train(self, test_train, filename, plot):
        df_raw = ppd.process_data(filename=filename)
        self.df_clean = ppd.clean_data(df_raw, plot=plot)

        if test_train != 'temporal':
            # Create a customer - product matrix (n x m)
            self.matrix, self.customers_map, self.products_map = ppd.create_customer_product_matrix(self.df_clean)
            self.n = len(self.customers_map)
            self.m = len(self.products_map)

        # Split the utility matrix into train and test data
        if test_train == 'one_out':
            self.train_matrix, self.test_data = split.leave_one_out(self.matrix, self.n)
        elif test_train == 'last_out':
            self.train_matrix, self.test_data = split.leave_last_out(self.matrix, self.df_clean, self.customers_map,
                                                                     self.products_map)
        elif test_train == 'temporal':
            self.matrix, self.customers_map, self.products_map, self.train_matrix, self.test_data, self.df_clean = split.temporal_split(
                self.df_clean)
            self.n = len(self.customers_map)
            self.m = len(self.products_map)

    def create_similarity_matrix(self):
        # For each customer, compute how similar they are to each other customer.
        if config.use_cupy:
            self.similarity_matrix = cosine_similarity(sparse.csr_matrix(np.asnumpy(self.matrix)))
        else:
            self.similarity_matrix = cosine_similarity(sparse.csr_matrix(self.matrix))
        self.similarity_matrix = (1 - self.alpha) * self.similarity_matrix + self.alpha * self.smd_matrix

    def predict_ratings_matrix(self):
        try:
            self.ratings_matrix = hf.read_matrix('basic_cf/ratings_matrix.csv')
        except IOError:
            if self.similarity_matrix is None:
                self.create_similarity_matrix()
            print("Didn't find ratings, creating it...")
            self.ratings_matrix = np.zeros((self.n, self.m))
            for i in range(self.n):
                k_neighbours = self.find_k_n_n(i)
                for j in range(self.m):
                    self.ratings_matrix[i][j] = self.compute_score(k_neighbours, j)

            # Set each rating to 0 for products which have already been bought.
            non_zero = np.where(self.matrix > 0)
            self.ratings_matrix[non_zero] = 0

            if self.save:
                hf.save_matrix(self.ratings_matrix, 'basic_cf/ratings_matrix.csv')

    def find_k_n_n(self, index):
        nearest_neighbours = np.argsort(self.similarity_matrix[index])[::-1]
        if len(nearest_neighbours) > self.k:
            return nearest_neighbours[1:self.k + 1]
        else:
            return nearest_neighbours[1:]

    def compute_score(self, neighbours, product):
        total = 0
        for n in neighbours:
            total += self.train_matrix[n][product]

        return total / self.k

    def predict_rating(self, i, j):
        k_neighbours = self.find_k_n_n(i)
        return self.compute_score(k_neighbours, j)

    def fit(self, fast=True):
        self.create_similarity_matrix()
        if fast:
            self.predict_ratings_matrix_fast()
        else:
            self.predict_ratings_matrix()

    # Optimized algorithm, using some clever linear algebra
    def predict_ratings_matrix_fast(self, bought_before=True):
        customers_filter = (np.argsort(np.argsort(self.similarity_matrix, axis=1)) >=
                            self.similarity_matrix.shape[
                                1] - self.k) * 1
        np.fill_diagonal(customers_filter, 0)

        self.ratings_matrix = (customers_filter @ self.train_matrix) / self.k

        if bought_before:
            # Set each rating to 0 for products which have already been bought.
            nonzero = np.where(self.train_matrix == 1, 0, 1)
            self.ratings_matrix = self.ratings_matrix * nonzero


def predict_recommendation(ratings_matrix, r):
    # Get the r highest ratings for each customer.
    recommendations = np.fliplr(np.argsort(ratings_matrix))[:, :min(r, len(ratings_matrix))]
    return recommendations
