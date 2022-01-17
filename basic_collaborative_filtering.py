import pathlib

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import test_model
import helper_functions as hf
import pre_process_data as ppd
import similarity_meta_data as smd
import split_data as split
import time


class CollaborativeFilteringBasic(object):
    def __init__(self, filename, split_method, k, alpha, plot, save):
        # Create the /basic_cf directory
        pathlib.Path('./data/basic_cf').mkdir(parents=True, exist_ok=True)

        self.k = k
        self.alpha = alpha
        self.save = save
        df_raw = ppd.process_data(filename=filename)
        self.df_clean = ppd.clean_data(df_raw, plot=plot)

        if split_method != 'temporal':
            # Create a customer - product matrix (n x m)
            self.matrix, self.customers_map, self.products_map = ppd.create_customer_product_matrix(self.df_clean)
            self.n = len(self.customers_map)
            self.m = len(self.products_map)

        # Split the utility matrix into train and test data
        if split_method == 'one_out':
            self.train_matrix, self.test_data = split.leave_one_out(self.matrix, self.n)
        elif split_method == 'last_out':
            self.train_matrix, self.test_data = split.leave_last_out(self.matrix, self.df_clean, self.customers_map,
                                                                     self.products_map)
        elif split_method == 'temporal':
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


def predict_recommendation(ratings_matrix, n, r):
    recommendations = np.zeros((n, r))
    # For each customer, find k nearest neighbours and predict r recommendations.
    for i in range(n - 1):
        ratings = np.argsort(ratings_matrix[i, :])[::-1]
        if len(ratings) > r:
            recommendations[i] = ratings[0:r]
        else:
            recommendations[i] = ratings

    return recommendations


# The idea of this method is to solve the challenge by reducing it
# to an instance of a collaborative filtering problem, with binary, positive only data.
# There are multiple algorithms which can be used to do this, the most basic one will be using
# a cosine similarity where the most-frequent product will be recommended.
# A more complex method will be using k-Unified Nearest Neighbours (k-UNN)
# Another method which will be explored is Alternating Least Squares.
def main():
    # Process data
    filename_xslx = './data/data-raw.xlsx'
    save = True

    # k nearest neighbours, r recommendations, alpha is how much we weigh the meta data similarity matrix
    k = 100
    r = 500
    alpha = 0.1

    start = time.time()

    cf_knn = CollaborativeFilteringBasic(filename_xslx, split_method='temporal', k=k,
                                         alpha=alpha, plot=False, save=save)
    cf_knn.create_similarity_matrix()
    cf_knn.predict_ratings_matrix()
    acc = test_model.hit_rate(model=cf_knn, r=r)

    print(time.time() - start)
    print(acc)


if __name__ == '__main__':
    main()
