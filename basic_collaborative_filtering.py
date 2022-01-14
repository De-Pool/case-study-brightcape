import numpy as np
from scipy import spatial

import helper_functions as hf
import pre_process_data as ppd
import split_data as split


# The idea of this method is to solve the challenge by reducing it
# to an instance of a collaborative filtering problem, with binary, positive only data.
# There are multiple algorithms which can be used to do this, the most basic one will be using
# a cosine similarity where the most-frequent product will be recommended.
# A more complex method will be using k-Unified Nearest Neighbours (k-UNN)
# Another method which will be explored is Alternating Least Squares.
def main():
    # Process data
    filename = './data/data-raw.xlsx'
    df_raw = ppd.process_data(filename=filename)

    # Clean data
    df_clean = ppd.clean_data(df_raw, False)

    # Create a customer - product matrix (n x m)
    matrix, customers_map, products_map = ppd.create_customer_product_matrix(df_clean)
    n = len(customers_map)
    m = len(products_map)

    # Split the utility matrix into train and test data
    train_matrix, test_data = split.leave_one_out(matrix, n)
    train_matrix, test_data = split.leave_last_out(matrix,
                                                   split.get_indices_last(df_clean, customers_map, products_map), n)

    # Create customer - customer similarity matrix (n x n)
    similarity_matrix = create_similarity_matrix(train_matrix, n)

    # k nearest neighbours -> based on customer similarity
    save = True
    k = 25
    ratings_matrix = predict_ratings_matrix(matrix, similarity_matrix, n, m, k, save)

    # r recommendations
    r = 10
    filename = 'recommendations.csv'
    recommendations = predict_recommendation(ratings_matrix, len(customers_map), r, filename, save)


def create_similarity_matrix(c_p_matrix, n):
    try:
        similarity_matrix = hf.read_matrix('similarity_matrix.csv')
    except IOError:
        print("Didn't find a similarity matrix, creating it...")
        similarity_matrix = np.zeros((n, n))

        # For each customer, compute how similar they are to each other customer.
        for i in range(n):
            for j in range(n):
                # cosine similarity = 1 - cosine distance
                similarity_matrix[i][j] = 1 - spatial.distance.cosine(c_p_matrix[i], c_p_matrix[j])

        hf.save_matrix(similarity_matrix, 'similarity_matrix.csv')

    return similarity_matrix


def predict_ratings_matrix(c_p_matrix, similarity_matrix, n, m, k, save):
    try:
        ratings_matrix = hf.read_matrix('ratings_matrix.csv')
    except IOError:
        print("Didn't find ratings, creating it...")
        ratings_matrix = np.zeros((n, m))
        for i in range(n):
            k_neighbours = find_k_n_n(i, similarity_matrix, k)
            for j in range(m):
                ratings_matrix[i][j] = compute_score(c_p_matrix, k_neighbours, j, k)

        # Set each rating to 0 for products which have already been bought.
        non_zero = np.where(c_p_matrix > 0)
        for i in range(len(non_zero[0])):
            ratings_matrix[non_zero[i]][non_zero[i]] = 0

        if save:
            hf.save_matrix(ratings_matrix, 'ratings_matrix.csv')

    return ratings_matrix


def predict_recommendation(ratings_matrix, n, r, filename, save):
    try:
        recommendations = hf.read_matrix(filename)
    except IOError:
        print("Didn't find recommendations, creating it...")
        recommendations = np.zeros((n, r))
        # For each customer, find k nearest neighbours and predict r recommendations.
        for i in range(n - 1):
            ratings = np.argsort(ratings_matrix[i, :])[::-1]
            if len(ratings) > r:
                recommendations[i] = ratings[0:r]
            else:
                recommendations[i] = ratings
        if save:
            hf.save_matrix(recommendations, filename)

    return recommendations


def find_k_n_n(index, similarity_matrix, k):
    nearest_neighbours = np.argsort(similarity_matrix[index])[::-1]
    if len(nearest_neighbours) > k:
        return nearest_neighbours[1:k + 1]
    else:
        return nearest_neighbours[1:]


def compute_score(c_p_matrix, neighbours, product, k):
    total = 0
    for n in neighbours:
        total += c_p_matrix[n][product]

    return total / k


if __name__ == '__main__':
    main()
