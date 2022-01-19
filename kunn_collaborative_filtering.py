import pathlib
from math import sqrt

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
import helper_functions as hf
import pre_process_data as ppd
import similarity_meta_data as smd
import split_data as split


class CollaborativeFilteringKUNN(object):
    def __init__(self, filename, split_method, k_products, k_customers, alpha, plot, save):
        if save:
            # Create the /kunn_cf directory
            pathlib.Path('./data/kunn_cf').mkdir(parents=True, exist_ok=True)

        self.alpha = alpha
        self.k_products = k_products
        self.k_customers = k_customers
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

        # c(x) is a function, which returns the count of x, wrt the rating
        # So given product p, c(p) will return the count of how much customers prefer product p
        self.c_products = self.train_matrix.sum(axis=0)
        self.c_customers = self.train_matrix.sum(axis=1)

        self.similarity_matrix_products = None
        self.similarity_matrix_customers = None
        self.ratings_matrix = None

    def create_similarity_matrices(self):
        try:
            self.similarity_matrix_products = hf.read_matrix('/kunn_cf/similarity_matrix_products.csv')
            self.similarity_matrix_customers = hf.read_matrix('/kunn_cf/similarity_matrix_customers.csv')
            self.similarity_matrix_customers = (
                                                       1 - self.alpha) * self.similarity_matrix_customers + self.alpha * self.smd_matrix
        except IOError:
            print("Didn't find similarity matrices, creating it...")

            # Create 2 similarity matrices, for both products and customers
            self.similarity_matrix_products = np.zeros((self.m, self.m))
            self.similarity_matrix_customers = np.zeros((self.n, self.n))

            # For each customer, compute how similar they are to each other customer.
            for i in range(self.n):
                for j in range(self.n):
                    self.similarity_matrix_customers[i][j] = self.customer_similarity(i, j)
            np.fill_diagonal(self.similarity_matrix_customers, 0)

            # For each product, compute how similar they are to each other product.
            for i in range(self.m):
                for j in range(self.m):
                    self.similarity_matrix_products[i][j] = self.product_similarity(i, j)
            np.fill_diagonal(self.similarity_matrix_products, 0)

            hf.save_matrix(self.similarity_matrix_products, '/kunn_cf/similarity_matrix_products.csv')
            hf.save_matrix(self.similarity_matrix_customers, '/kunn_cf/similarity_matrix_customers.csv')

    def predict_ratings_matrix(self):
        try:
            self.ratings_matrix = hf.read_matrix('kunn_cf/ratings_matrix.csv')
        except IOError:
            if self.similarity_matrix_products is None or self.similarity_matrix_customers is None:
                self.create_similarity_matrices()

            print("Didn't find ratings, creating it...")
            self.ratings_matrix = np.zeros((self.n, self.m))

            for i in range(self.n):
                k_neighbours_customer = self.find_k_n_n_customer(i)
                for j in range(self.m):
                    k_neighbours_product = self.find_k_n_n_product(j)
                    rating_c = self.compute_rating_customer(j, k_neighbours_customer)
                    rating_p = self.compute_rating_product(i, k_neighbours_product)
                    self.ratings_matrix[i][j] = rating_c + rating_p

                    # Set each rating to 0 for products which have already been bought.
            non_zero = np.where(self.matrix > 0)
            self.ratings_matrix[non_zero] = 0

            if self.save:
                hf.save_matrix(self.ratings_matrix, 'kunn_cf/ratings_matrix.csv')

    def customer_similarity(self, i, j):
        similarity = 0
        for p in range(self.m):
            if self.c_customers[i] * self.c_customers[j] != 0 and self.c_products[p] != 0:
                n = self.train_matrix[i][p] * self.train_matrix[j][p]
                d = sqrt(self.c_customers[i] * self.c_customers[j] * self.c_products[p])
                similarity += n / d
        return similarity

    def product_similarity(self, i, j):
        similarity = 0
        for c in range(self.n):
            if self.c_products[i] * self.c_products[j] != 0 and self.c_customers[c] != 0:
                n = self.train_matrix[c][i] * self.train_matrix[c][j]
                d = sqrt(self.c_products[i] * self.c_products[j] * self.c_customers[c])
                similarity += n / d
        return similarity

    def compute_rating_customer(self, j, knn_customer):
        if self.c_products[j] == 0:
            return 0
        rating = 0
        for c in knn_customer:
            rating += (self.train_matrix[c][j] * self.similarity_matrix_customers[c][j]) / sqrt(self.c_products[j])

        return rating

    def compute_rating_product(self, i, knn_product):
        if self.c_customers[i] == 0:
            return 0
        rating = 0
        for p in knn_product:
            rating += (self.train_matrix[i][p] * self.similarity_matrix_products[i][p]) / sqrt(self.c_customers[i])
        return rating

    def find_k_n_n_customer(self, i):
        nearest_neighbours = np.argsort(self.similarity_matrix_customers[i])[::-1]
        if len(nearest_neighbours) > self.k_customers:
            return nearest_neighbours[1:self.k_customers + 1]
        else:
            return nearest_neighbours[1:]

    def find_k_n_n_product(self, i):
        nearest_neighbours = np.argsort(self.similarity_matrix_products[i])[::-1]
        if len(nearest_neighbours) > self.k_products:
            return nearest_neighbours[1:self.k_products + 1]
        else:
            return nearest_neighbours[1:]

    def fit(self, fast=True, normalize=True):
        if fast:
            self.create_similarity_matrices_fast()
            self.predict_ratings_matrix_fast(normalize)
        else:
            self.create_similarity_matrices()
            self.predict_ratings_matrix()

    # Optimized algorithms, using some clever linear algebra
    def create_similarity_matrices_fast(self):
        # Compute product - product similarity matrix
        c_products_sqrt = (1 / np.sqrt(self.c_products)).reshape((len(self.c_products), 1))
        a = c_products_sqrt @ c_products_sqrt.T
        b = self.train_matrix.T @ np.diag(1 / np.sqrt(self.c_customers)) @ self.train_matrix
        self.similarity_matrix_products = np.multiply(a, b)
        np.fill_diagonal(self.similarity_matrix_products, 0)

        # Compute customer - customer similarity matrix
        c_customers_sqrt = (1 / np.sqrt(self.c_customers)).reshape((len(self.c_customers), 1))
        a = c_customers_sqrt @ c_customers_sqrt.T
        b = self.train_matrix @ np.diag(1 / np.sqrt(self.c_products)) @ self.train_matrix.T
        self.similarity_matrix_customers = np.multiply(a, b)
        np.fill_diagonal(self.similarity_matrix_customers, 0)

    def predict_ratings_matrix_fast(self, normalize):
        # Only use k nearest neighbours
        products_filter = (np.argsort(np.argsort(self.similarity_matrix_products, axis=1)) >=
                           self.similarity_matrix_products.shape[1] - self.k_products)
        customers_filter = (np.argsort(np.argsort(self.similarity_matrix_customers, axis=0)) >=
                            self.similarity_matrix_customers.shape[
                                1] - self.k_customers)

        products_filtered = self.similarity_matrix_products * products_filter
        customers_filtered = self.similarity_matrix_customers * customers_filter

        product_ratings = self.train_matrix @ products_filtered
        customer_ratings = customers_filtered @ self.train_matrix
        if normalize:
            product_ratings = np.diag(1.0 / np.sqrt(self.c_customers)) @ product_ratings
            customer_ratings = customer_ratings @ np.diag(1.0 / np.sqrt(self.c_products))

        self.ratings_matrix = product_ratings + customer_ratings

        # Set each rating to 0 for products which have already been bought.
        non_zero = np.where(self.matrix > 0)
        self.ratings_matrix[non_zero] = 0
