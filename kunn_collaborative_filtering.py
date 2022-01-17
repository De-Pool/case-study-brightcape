from math import sqrt

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
import basic_collaborative_filtering as bcf
import helper_functions as hf
import pre_process_data as ppd
import similarity_meta_data as smd
import split_data as split


class CollaborativeFilteringKUNN(object):
    def __init__(self, filename, split_method, k_products, k_customers, plot):
        self.k_products = k_products
        self.k_customers = k_customers

        df_raw = ppd.process_data(filename=filename)
        self.df_clean = ppd.clean_data(df_raw, plot=plot)

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
        else:
            self.train_matrix, self.test_data = self.matrix, []

        # Create customer - customer meta data similarity matrix (n x n)
        self.smd_matrix = smd.meta_data_similarity_matrix(self.df_clean, self.customers_map, self.n)

        self.c_products = None
        self.c_customers = None
        self.similarity_matrix_products = None
        self.similarity_matrix_customers = None
        self.ratings_matrix = None

    def create_similarity_matrices(self):
        try:
            self.similarity_matrix_products = hf.read_matrix('/kunn_cf/similarity_matrix_products.csv')
            self.similarity_matrix_customers = hf.read_matrix('/kunn_cf/similarity_matrix_customers.csv')
        except IOError:
            print("Didn't find similarity matrices, creating it...")

            # Create 2 similarity matrices, for both products and customers
            self.similarity_matrix_products = np.zeros((self.m, self.m))
            self.similarity_matrix_customers = np.zeros((self.m, self.m))

            # c(x) is a function, which returns the count of x, wrt the rating
            # So given product p, c(p) will return the count of how much customers prefer product p
            self.c_products = self.train_matrix.sum(axis=0)
            self.c_customers = self.train_matrix.sum(axis=1)

            # For each customer, compute how similar they are to each other customer.
            for i in range(self.n):
                for j in range(self.n):
                    self.similarity_matrix_customers[i][j] = self.customer_similarity(i, j)

            # For each product, compute how similar they are to each other product.
            for i in range(self.m):
                for j in range(self.m):
                    self.similarity_matrix_products[i][j] = self.product_similarity(i, j)

            hf.save_matrix(self.similarity_matrix_products, '/kunn_cf/similarity_matrix_products.csv')
            hf.save_matrix(self.similarity_matrix_customers, '/kunn_cf/similarity_matrix_customers.csv')

    def predict_ratings_matrix(self, save):
        try:
            self.ratings_matrix = hf.read_matrix('kunn_cf/ratings_matrix.csv')
        except IOError:
            if self.c_products is None or self.c_customers is None \
                    or self.similarity_matrix_products is None or self.similarity_matrix_customers is None:
                self.create_similarity_matrices()

            print("Didn't find ratings, creating it...")
            self.ratings_matrix = np.zeros((self.n, self.m))

            for i in range(self.n):
                k_neighbours_customer = self.find_k_n_n_customer(i)
                for j in range(self.m):
                    k_neighbours_product = self.find_k_n_n_product(j)
                    self.ratings_matrix[i][j] = self.compute_score(i, j, k_neighbours_customer, k_neighbours_product)

            # Set each rating to 0 for products which have already been bought.
            non_zero = np.where(self.matrix > 0)
            for i in range(len(non_zero[0])):
                self.ratings_matrix[non_zero[i]][non_zero[i]] = 0

            if save:
                hf.save_matrix(self.ratings_matrix, 'kunn_cf/ratings_matrix.csv')

    def customer_similarity(self, i, j):
        # Get all products, both customer i and customer j prefer
        in_common_products = np.intersect1d(np.nonzero(self.train_matrix[i, :])[0],
                                            np.nonzero(self.train_matrix[j, :])[0])
        # If customer i and customer j have no products in common, then they have a similarity of 0.
        if len(in_common_products) == 0:
            return 0
        else:
            customer_product = self.c_customers[i] * self.c_customers[j]
            similarity = 0
            for product in in_common_products:
                similarity += 1 / sqrt(customer_product * self.c_products[product])
            return similarity

    def product_similarity(self, i, j):
        # Get all customers, which prefer both product i and product j
        in_common_customers = np.intersect1d(np.nonzero(self.train_matrix[:, i])[0],
                                             np.nonzero(self.train_matrix[:, j])[0])
        # If product i and product j have no customers in common, then they have a similarity of 0.
        if len(in_common_customers) == 0:
            return 0
        else:
            product_product = self.c_products[i] * self.c_products[j]
            similarity = 0
            for customer in in_common_customers:
                similarity += 1 / sqrt(product_product * self.c_customers[customer])
            return similarity

    def compute_score(self, i, j, knn_customer, knn_product):
        # Get score which user i will give to item j
        score_customer = self.compute_score_customer(j, knn_customer)
        score_product = self.compute_score_product(i, knn_product)
        return score_product + score_customer

    def compute_score_customer(self, j, knn_customer):
        if self.c_products[j] == 0:
            return 0
        total = 0
        for user, sim in zip(knn_customer[0], knn_customer[1]):
            if self.train_matrix[user, j] > 0:
                total += sim

        return total / sqrt(self.c_products[j])

    def compute_score_product(self, i, knn_product):
        if self.c_customers[i] == 0:
            return 0
        total = 0
        for product, sim in zip(knn_product[0], knn_product[1]):
            if self.train_matrix[i, product] > 0:
                total += sim
        return total / sqrt(self.c_customers[i])

    def find_k_n_n_customer(self, i):
        # Find all similarities between customer i and every other customer.
        all_sims = self.similarity_matrix_customers[:, i]
        # Find the largest k_customers + 1 elements
        knn = np.argpartition(all_sims, -self.k_customers - 1)[-self.k_customers - 1:]
        # Remove the index of i if present then just in case it wasn't present keep only the largest k_customers.
        knn = knn[knn != i][-self.k_customers:]
        return knn, all_sims[knn]

    def find_k_n_n_product(self, i):
        # Find all similarities between product i and every other product.
        all_sims = self.similarity_matrix_products[:, i]
        # Find the largest k_products + 1 elements
        knn = np.argpartition(all_sims, -self.k_products - 1)[-self.k_products - 1:]
        # Remove the index of i if present then just in case it wasn't present keep only the largest k_products.
        knn = knn[knn != i][-self.k_products:]
        return knn, all_sims[knn]


# The idea of this method is to solve the challenge by reducing it
# to an instance of a collaborative filtering problem, with binary, positive only data.
# There are multiple algorithms which can be used to do this, the most basic one will be using
# a cosine similarity where the most-frequent product will be recommended.
# A more complex method will be using k-Unified Nearest Neighbours (k-UNN)
# Another method which will be explored is Alternating Least Squares.
def main():
    filename_xlsx = './data/data-raw.xlsx'
    filename_recommendations = '/kunn_cf/recommendations.csv'
    save = True
    # k nearest neighbours for customers
    k_c = 25
    # k_p nearest neighbours for products
    k_p = 25
    # Get r recommendations
    r = 10

    b_kunn = CollaborativeFilteringKUNN(filename_xlsx, 'last_out', k_p, k_c, False)

    # Create customer - customer similarity matrix (n x n) and a product - product similarity matrix (m x m)
    b_kunn.predict_ratings_matrix(save=False)

    recommendations = bcf.predict_recommendation(b_kunn.ratings_matrix, b_kunn.n, r, filename_recommendations, save)


if __name__ == '__main__':
    main()
