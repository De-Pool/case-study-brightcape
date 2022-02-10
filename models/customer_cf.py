import pathlib

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import helper_functions as hf
import similarity_meta_data as smd
import split_data as split
import test_model


class CollaborativeFilteringBasic(object):
    def __init__(self, filename, model_data, k, alpha=0, plot=False, save=False):
        self.k = k
        self.alpha = alpha
        self.save = save

        if save:
            # Create the /basic_cf directory
            pathlib.Path('../data/basic_cf').mkdir(parents=True, exist_ok=True)

        if not isinstance(model_data, dict):
            # data can also be: temporal, last_out, one_out
            model_data = split.create_model_data(filename, plot, model_data)

        self.matrix = model_data['matrix']
        self.train_matrix = model_data['train_matrix']
        self.customers_map = model_data['customers_map']
        self.products_map = model_data['products_map']
        self.test_data = model_data['test_data']
        self.df_clean = model_data['df_clean']
        self.n = model_data['n']
        self.m = model_data['m']

        # Create customer - customer meta data similarity matrix (n x n)
        self.smd_matrix = smd.meta_data_similarity_matrix(self.df_clean, self.customers_map, self.n)

        self.similarity_matrix = None
        self.ratings_matrix = None

    def create_similarity_matrix(self):
        # For each customer, compute how similar they are to each other customer.
        if config.use_cupy:
            self.similarity_matrix = np.asarray(cosine_similarity(sparse.csr_matrix(np.asnumpy(self.train_matrix))))
        else:
            self.similarity_matrix = cosine_similarity(sparse.csr_matrix(self.train_matrix))
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


def gridsearch(model_data, k_s, similar_items=False, similar_products_dict=None):
    best_basic = [0]
    best_param_basic = 0
    all_params_basic = []
    model_basic_cf = CollaborativeFilteringBasic('', model_data, 1, 0, False, False)
    model_basic_cf.fit()
    for k in k_s:
        model_basic_cf.k = k
        model_basic_cf.predict_ratings_matrix_fast()
        performance_basic = test_model.all_methods(model_basic_cf, model_data['r'], similar_items,
                                                   similar_products_dict)
        all_params_basic.append([k, performance_basic])
        if performance_basic[0] > best_basic[0]:
            best_basic = performance_basic
            best_param_basic = k
    return best_basic, best_param_basic, all_params_basic


def gridsearch_alpha(model_data, k, alphas, similar_items=False, similar_products_dict=None):
    best_basic = [0]
    best_param_basic = 0
    all_params_basic = []

    for alpha in alphas:
        model_basic_cf = CollaborativeFilteringBasic('', model_data, k, alpha=alpha)
        model_basic_cf.fit()
        performance_basic = test_model.all_methods(model_basic_cf, model_data['r'], similar_items,
                                                   similar_products_dict)
        all_params_basic.append([k, performance_basic])
        if performance_basic[0] > best_basic[0]:
            best_basic = performance_basic
            best_param_basic = alpha
    return best_basic, best_param_basic, all_params_basic
