import implicit

import basic_collaborative_filtering as bcf
import kunn_collaborative_filtering as kunn
import test_model


def main():
    filename_xlsx = './data/data-raw.xlsx'
    save = False
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
    r['train_matrix'] = model_kunn.train_matrix
    r['test_data'] = model_kunn.test_data
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
