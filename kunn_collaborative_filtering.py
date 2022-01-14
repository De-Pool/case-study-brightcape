import numpy as np
import helper_functions as hf
import basic_collaborative_filtering as cf
from math import sqrt


# The idea of this method is to solve the challenge by reducing it
# to an instance of a collaborative filtering problem, with binary, positive only data.
# There are multiple algorithms which can be used to do this, the most basic one will be using
# a cosine similarity where the most-frequent product will be recommended.
# A more complex method will be using k-Unified Nearest Neighbours (k-UNN)
# Another method which will be explored is Alternating Least Squares.
def main():
    # Process data
    filename = './data/data-raw.xlsx'
    df_raw = cf.process_data(filename=filename)

    # Clean data
    df_clean = cf.clean_data(df_raw, plot=False)

    # Create a customer - product matrix (n x m)
    matrix, customers_map, products_map = cf.create_customer_product_matrix(df_clean)

    n = len(customers_map)
    m = len(products_map)
    # Create customer - customer similarity matrix (n x n) and a product - product similarity matrix (m x m)
    similarity_matrix_customers, similarity_matrix_products = create_similarity_matrices(matrix, n, m)

    k_customer = 25
    k_product = 25
    ratings_matrix = predict_ratings_matrix(matrix, similarity_matrix_customers, similarity_matrix_products, n, m,
                                            k_customer, k_product)


def create_similarity_matrices(c_p_matrix, n, m):
    try:
        similarity_matrix_products = hf.read_matrix('/kunn_cf/similarity_matrix_products.csv')
        similarity_matrix_customers = hf.read_matrix('/kunn_cf/similarity_matrix_customers.csv')
    except IOError:
        print("Didn't find similarity matrices, creating it...")

        # Create 2 similarity matrices, for both products and customers
        similarity_matrix_products = np.zeros((m, m))
        similarity_matrix_customers = np.zeros((n, n))

        # c(x) is a function, which returns the count of x, wrt the rating
        # So given product p, c(p) will return the count of how much customers prefer product p
        c_products = c_p_matrix.sum(axis=0)
        c_customers = c_p_matrix.sum(axis=1)

        # For each customer, compute how similar they are to each other customer.
        for i in range(n):
            for j in range(n):
                similarity_matrix_customers[i][j] = customer_similarity(i, j, c_p_matrix, c_customers, c_products)

        # For each product, compute how similar they are to each other product.
        for i in range(m):
            for j in range(m):
                similarity_matrix_products[i][j] = product_similarity(i, j, c_p_matrix, c_customers, c_products)

        hf.save_matrix(similarity_matrix_products, '/kunn_cf/similarity_matrix_products.csv')
        hf.save_matrix(similarity_matrix_customers, '/kunn_cf/similarity_matrix_customers.csv')

    return similarity_matrix_customers, similarity_matrix_products


def customer_similarity(i, j, c_p_matrix, c_customers, c_products):
    # Get all products, both customer i and customer j prefer
    in_common_products = np.intersect1d(np.nonzero(c_p_matrix[i, :])[0], np.nonzero(c_p_matrix[j, :])[0])
    # If customer i and customer j have no products in common, then they have a similarity of 0.
    if len(in_common_products) == 0:
        return 0
    else:
        customer_product = c_customers[i] * c_customers[j]
        similarity = 0
        for product in in_common_products:
            similarity += 1 / sqrt(customer_product * c_products[product])
        return similarity


def product_similarity(i, j, c_p_matrix, c_customers, c_products):
    # Get all customers, which prefer both product i and product j
    in_common_customers = np.intersect1d(np.nonzero(c_p_matrix[:, i])[0], np.nonzero(c_p_matrix[:, j])[0])
    # If product i and product j have no customers in common, then they have a similarity of 0.
    if len(in_common_customers) == 0:
        return 0
    else:
        product_product = c_products[i] * c_products[j]
        similarity = 0
        for customer in in_common_customers:
            similarity += 1 / sqrt(product_product * c_customers[customer])
        return similarity


def predict_ratings_matrix(c_p_matrix, similarity_matrix_c, similarity_matrix_p, n, m, k_c, k_p):
    try:
        ratings_matrix = hf.read_matrix('kunn_cf/ratings_matrix.csv')
    except IOError:
        print("Didn't find ratings, creating it...")
        ratings_matrix = np.zeros((n, m))
        # c(x) is a function, which returns the count of x, wrt the rating
        # So given product p, c(p) will return the count of how much customers prefer product p
        c_products = c_p_matrix.sum(axis=0)
        c_customers = c_p_matrix.sum(axis=1)
        for i in range(n):
            k_neighbours_customer = find_k_n_n_customer(i, similarity_matrix_c, k_c)
            for j in range(m):
                k_neighbours_product = find_k_n_n_product(j, similarity_matrix_p, k_p)
                c_p_matrix[i][j] = compute_score(i, j, c_p_matrix, k_neighbours_customer, k_neighbours_product,
                                                 c_products, c_customers)

        hf.save_matrix(ratings_matrix, 'ratings_matrix.csv')

    return ratings_matrix


def compute_score(i, j, c_p_matrix, knn_customer, knn_product, c_products, c_customers):
    # Get score which user u will give to item i
    total = 0
    for user, sim in zip(knn_customer[0], knn_customer[1]):
        if c_p_matrix[user, j] > 0:
            total += sim
    score_customer = total / sqrt(c_products[j])

    total = 0
    for product, sim in zip(knn_product[0], knn_product[1]):
        if c_p_matrix[i, product] > 0:
            total += sim
    score_product = total / sqrt(c_customers[i])

    return score_product + score_customer


def find_k_n_n_customer(i, similarity_matrix, k):
    # Find all similarities between customer i and every other customer.
    all_sims = similarity_matrix[:, i]
    # Find the largest kI + 1 elements
    knn = np.argpartition(all_sims, -k - 1)[-k - 1:]
    # Remove the index of i if present then just in case it wasn't present keep only the largest k.
    knn = knn[knn != i][-k:]
    return knn, all_sims[knn]


def find_k_n_n_product(i, similarity_matrix, k):
    # Find all similarities between customer i and every other customer.
    all_sims = similarity_matrix[:, i]
    # Find the largest kI + 1 elements
    knn = np.argpartition(all_sims, -k - 1)[-k - 1:]
    # Remove the index of i if present then just in case it wasn't present keep only the largest kU.
    knn = knn[knn != i][-k:]
    return knn, all_sims[knn]


if __name__ == '__main__':
    main()
