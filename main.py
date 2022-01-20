from implicit.approximate_als import AlternatingLeastSquares as ALS
from implicit.lmf import LogisticMatrixFactorization as LMF
from implicit.bpr import BayesianPersonalizedRanking as BPR

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
    k_c = 50
    # k_p nearest neighbours for products
    k_p = 50
    # Get r recommendations
    recommendations = 250
    alpha = 0
    split_method = 'last_out'

    model_basic_cf = bcf.CollaborativeFilteringBasic(filename_xlsx, split_method, k_c, alpha, False, save)
    model_basic_cf.fit()
    recommendations = bcf.predict_recommendation(model_basic_cf.ratings_matrix, recommendations)
    test_model.find_similar_items(model_basic_cf.test_data, recommendations, model_basic_cf.products_map, model_basic_cf.df_clean)

    performance_basic = test_model.all_methods(model_basic_cf, recommendations)

    model_kunn = kunn.CollaborativeFilteringKUNN(filename_xlsx, split_method, k_p, k_c, alpha, False, save)
    model_kunn.fit()
    performance_kunn = test_model.all_methods(model_kunn, recommendations)

    r = {'train_matrix': model_kunn.train_matrix, 'test_data': model_kunn.test_data, 'r': recommendations,
         'split_method': split_method}

    performance_ALS = test_model.all_methods(ALS(), r)
    # performance_bpr = test_model.all_methods(BPR(), r)
    # performance_lmf = test_model.all_methods(LMF(), r)

    print("kunn", performance_kunn)
    print("basic", performance_basic)
    print("als", performance_ALS)
    # print("bpr", performance_bpr)
    # print("lmf", performance_lmf)


if __name__ == '__main__':
    main()
