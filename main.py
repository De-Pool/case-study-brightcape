import implicit

import basic_collaborative_filtering as bcf
import kunn_collaborative_filtering as kunn
import test_model


# The idea of this method is to solve the challenge by reducing it
# to an instance of a collaborative filtering problem, with binary, positive only data.
# There are multiple algorithms which can be used to do this, the most basic one will be using
# a cosine similarity where the most-frequent product will be recommended.
# A more complex method will be using k-Unified Nearest Neighbours (k-UNN)
# Another method which will be explored is Alternating Least Squares.
def main():
    filename_xlsx = './data/data-raw.xlsx'
    save = True
    # k nearest neighbours for customers
    k_c = 30
    # k_p nearest neighbours for products
    k_p = 30
    # Get r recommendations
    recommendations = 250
    alpha = 0.1
    split_method = 'last_out'

    model_basic_cf = bcf.CollaborativeFilteringBasic(filename_xlsx, split_method, 50, alpha, False, save)
    model_basic_cf.fit()

    model_kunn = kunn.CollaborativeFilteringKUNN(filename_xlsx, split_method, k_p, k_c, alpha, False, save)
    model_kunn.fit()

    # ALS
    r = dict()
    r['train_matrix'] = model_basic_cf.train_matrix
    r['test_data'] = model_basic_cf.test_data
    r['r'] = recommendations

    model_als = implicit.als.AlternatingLeastSquares(factors=50)

    # Compute hit rates
    hit_rate_als = test_model.hit_rate(model_als, r)
    hit_rate_kunn = test_model.hit_rate(model_kunn, recommendations)
    hit_rate_basic = test_model.hit_rate(model_basic_cf, recommendations)

    print(hit_rate_als)
    print(hit_rate_kunn)
    print(hit_rate_basic)


if __name__ == '__main__':
    main()
