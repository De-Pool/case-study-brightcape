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
    def __init__(self, filename, split_method, k, alpha, plot, save):
        if save:
            # Create the /basic_cf directory
            pathlib.Path('./data/basic_cf').mkdir(parents=True, exist_ok=True)

        self.k = k
        self.alpha = alpha
        self.save = save
        self.split_method = split_method

        df_raw = ppd.process_data(filename=filename)
        self.df_clean = ppd.clean_data(df_raw, plot=plot)

        if self.split_method != 'temporal':
            # Create a customer - product matrix (n x m)
            self.matrix, self.customers_map, self.products_map = ppd.create_customer_product_matrix(self.df_clean)
            self.n = len(self.customers_map)
            self.m = len(self.products_map)

        # Split the utility matrix into train and test data
        if self.split_method == 'one_out':
            self.train_matrix, self.test_data = split.leave_one_out(self.matrix, self.n)
        elif self.split_method == 'last_out':
            self.train_matrix, self.test_data = split.leave_last_out(self.matrix, self.df_clean, self.customers_map,
                                                                     self.products_map)
        elif self.split_method == 'temporal':
            self.matrix, self.customers_map, self.products_map, self.train_matrix, self.test_data, self.df_clean = split.temporal_split(
                self.df_clean, 0.05)
            self.n = len(self.customers_map)
            self.m = len(self.products_map)
        else:
            self.train_matrix, self.test_data = self.matrix, []

        # Create customer - customer meta data similarity matrix (n x n)
        self.smd_matrix = smd.meta_data_similarity_matrix(self.df_clean, self.customers_map, self.n)

        self.similarity_matrix = None
        self.ratings_matrix = None

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

    def fit(self):
        self.create_similarity_matrix()
        self.predict_ratings_matrix()


def predict_recommendation(ratings_matrix, n, r):
    recommendations = np.zeros((n, r))
    # For each customer, predict r recommendations.
    for i in range(n - 1):
        ratings = np.argsort(ratings_matrix[i, :])[::-1]
        if len(ratings) > r:
            recommendations[i] = ratings[0:r]
        else:
            recommendations[i] = ratings

    return recommendations
