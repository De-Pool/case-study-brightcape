import config
import helper_functions
import split_data

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from implicit.approximate_als import AlternatingLeastSquares as ALS

from helper_functions import plot
from helper_functions import plot3
from helper_functions import create_x_y

import models.product_cf as pcf
import models.customer_cf as ccf
import models.kunn_cf as kunn
import models.kunn_custom_cf as kunn_custom
import test_model as test


def main():
    filename_xlsx = './data/data-raw.xlsx'
    # Split method can be: temporal, one_out, last_out
    split_method = 'temporal'
    amount_recommendations = 250
    alpha = 0.05
    similar_items = False
    recommendations = np.arange(1, 50, 1)

    # best_models(split_method, amount_recommendations, alpha, similar_items, filename_xlsx)
    find_best_parameters(split_method, amount_recommendations, alpha, similar_items, filename_xlsx)
    varied_test_size(amount_recommendations, similar_items, filename_xlsx)
    varied_recommendations(split_method, alpha, similar_items, recommendations, filename_xlsx)

    # Train best model and generate the final deliverable
    model_data = split_data.create_model_data('./data/data-raw.xlsx', split_method='temporal', alpha=0)
    create_recommendations(kunn.CollaborativeFilteringKUNN('', model_data, k_products=80, k_customers=0),
                           amount_recommendations=50)


def varied_test_size(recommendations, similar_items, filename_xlsx):
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35,
              0.4]

    y_hit_rate = []
    y_ncdg = []
    y_map = []

    for alpha in alphas:
        model_data = split_data.create_model_data(filename_xlsx, split_method='temporal', alpha=alpha,
                                                  r=recommendations)
        similar_products_dict = test.compute_similar_products(model_data['df_clean'])

        model_basic_cf = ccf.CollaborativeFilteringBasic('', model_data, k=200)
        model_basic_cf.fit()

        model_product_cf = pcf.CollaborativeFilteringProduct('', model_data, k=40)
        model_product_cf.fit()

        model_kunn = kunn.CollaborativeFilteringKUNN('', model_data, k_products=80, k_customers=0)
        model_kunn.fit()

        performance_basic = test.all_methods(model_basic_cf, recommendations, similar_items,
                                             similar_products_dict)
        performance_product_cf = test.all_methods(model_product_cf, recommendations, similar_items,
                                                  similar_products_dict)
        performance_kunn = test.all_methods(model_kunn, recommendations, similar_items,
                                            similar_products_dict)
        performance_ALS = test.all_methods(ALS(factors=30, iterations=50), model_data, similar_items,
                                           similar_products_dict)
        y_hit_rate.append([performance_basic[0], performance_product_cf[0], performance_kunn[0], performance_ALS[0]])
        y_ncdg.append([performance_basic[1], performance_product_cf[1], performance_kunn[1], performance_ALS[1]])
        y_map.append([performance_basic[2], performance_product_cf[2], performance_kunn[2], performance_ALS[2]])

    labels = ['Customer CF', 'Product CF', 'k-UNN', 'ALS']
    plot(alphas, np.array(y_hit_rate).T, labels, 'Alpha', 'Hit rate',
         'Performance of models with varied test set size')
    plot(alphas, np.array(y_ncdg).T, labels, 'Alpha', 'NCDG',
         'Performance of models with varied test set size')
    plot(alphas, np.array(y_map).T, labels, 'Alpha', 'MAP',
         'Performance of models with varied test set size')


def varied_recommendations(split_method, alpha, similar_items, rs, filename_xlsx):
    data = split_data.create_model_data(filename_xlsx, split_method, alpha)
    similar_products_dict = test.compute_similar_products(data['df_clean'])

    model_basic_cf = ccf.CollaborativeFilteringBasic('', data, k=200)
    model_basic_cf.fit()

    model_product_cf = pcf.CollaborativeFilteringProduct('', data, k=40)
    model_product_cf.fit()

    k_c = 0
    k_p = 80
    model_kunn = kunn.CollaborativeFilteringKUNN('', data, k_p, k_c)
    model_kunn.fit()

    y_hit_rate = []
    y_ncdg = []
    y_map = []
    for r in rs:
        data['r'] = r
        performance_basic = test.all_methods(model_basic_cf, r, similar_items, similar_products_dict)
        performance_product_cf = test.all_methods(model_product_cf, r, similar_items, similar_products_dict)
        performance_kunn = test.all_methods(model_kunn, r, similar_items, similar_products_dict)
        performance_ALS = test.all_methods(ALS(factors=30, iterations=50), data, similar_items,
                                           similar_products_dict)

        y_hit_rate.append([performance_basic[0], performance_product_cf[0], performance_kunn[0], performance_ALS[0]])
        y_ncdg.append([performance_basic[1], performance_product_cf[1], performance_kunn[1], performance_ALS[1]])
        y_map.append([performance_basic[2], performance_product_cf[2], performance_kunn[2], performance_ALS[2]])

    labels = ['Customer CF', 'Product CF', 'k-UNN', 'ALS']
    plot(rs, np.array(y_hit_rate).T, labels, 'Recommendations', 'Hit rate',
         'Performance of models with varied amounts of recommendations')
    plot(rs, np.array(y_ncdg).T, labels, 'Recommendations', 'NCDG',
         'Performance of models with varied amounts of recommendations')
    plot(rs, np.array(y_map).T, labels, 'Recommendations', 'MAP',
         'Performance of models with varied amounts of recommendations')


def best_models(split_method, recommendations, alpha, similar_items, filename_xlsx):
    data = split_data.create_model_data(filename_xlsx, split_method, alpha)
    similar_products_dict = test.compute_similar_products(data['df_clean'])

    k_c_b = 200
    model_basic_cf = ccf.CollaborativeFilteringBasic(filename_xlsx, data, k_c_b)
    model_basic_cf.fit()
    performance_basic = test.all_methods(model_basic_cf, recommendations, similar_items,
                                         similar_products_dict)

    k_p_b = 40
    model_product_cf = pcf.CollaborativeFilteringProduct(filename_xlsx, data, k_p_b)
    model_product_cf.fit()
    performance_product_cf = test.all_methods(model_product_cf, recommendations, similar_items,
                                              similar_products_dict)

    k_c_custom = 180
    k_p_custom = 10
    model_kunn_custom = kunn_custom.CollaborativeFilteringKUNNCustom(filename_xlsx, data, k_c_custom, k_p_custom)
    model_kunn_custom.fit()
    performance_kunn_custom = test.all_methods(model_kunn_custom, recommendations, similar_items,
                                               similar_products_dict)

    # k nearest neighbours for customers
    k_c = 0
    # k_p nearest neighbours for products
    k_p = 80
    model_kunn = kunn.CollaborativeFilteringKUNN(filename_xlsx, data, k_p, k_c)
    model_kunn.fit()
    performance_kunn = test.all_methods(model_kunn, recommendations, similar_items,
                                        similar_products_dict)

    factors = 30
    iterations = 50
    performance_ALS = test.all_methods(ALS(factors=factors, iterations=iterations), data, similar_items,
                                       similar_products_dict)

    print("basic customer", performance_basic)
    print("basic product", performance_product_cf)
    print("kunn", performance_kunn)
    print("kunn custom", performance_kunn_custom)
    print("als", performance_ALS)


def find_best_parameters(split_method, recommendations, alpha, similar_items, filename_xlsx):
    data = split_data.create_model_data(filename_xlsx, split_method, alpha=alpha, r=recommendations)
    similar_products_dict = test.compute_similar_products(data['df_clean'])
    performance_measures = ['Hit rate', 'NCDG', 'MAP']

    # Hyper parameters k-UNN
    k_cs = np.arange(0, 200, 10)
    k_cs[0] = 1
    k_ps = np.arange(0, 200, 10)
    k_ps[0] = 1

    # Hyper parameters basic CF
    k_s_product = np.arange(0, 200, 10)
    k_s_product[0] = 1
    k_s_customer = np.arange(0, 200, 10)
    k_s_customer[0] = 1

    # Hyper parameters ALS
    factors = np.arange(0, 200, 10)
    factors[0] = 1
    iterations = np.arange(0, 200, 10)
    iterations[0] = 1

    # # k_c = 140, k_p = 180
    # best, best_params, all_params = kunn_custom.gridsearch(data, k_cs, [180], similar_items, similar_products_dict)
    # x_customer_kunn_c, y_customer_kunn_c = create_x_y(all_params, 2, 0)
    # plot3(x_customer_kunn_c, y_customer_kunn_c, performance_measures, 'k-customer',
    #       'Performance of k-UNN custom collaborative filtering')
    #
    # best, best_params, all_params = kunn_custom.gridsearch(data, [10], k_ps, similar_items, similar_products_dict)
    # x_product_kunn_c, y_product_kunn_c = create_x_y(all_params, 2, 1)
    # plot3(x_product_kunn_c, y_product_kunn_c, performance_measures, 'k-product',
    #       'Performance of k-UNN custom collaborative filtering')

    # Find best alpha when using the meta data similarity matrix
    # alphas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35,
    #           0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alphas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2]
    best_basic, best_param_basic, all_params_basic = ccf.gridsearch_alpha(data, 200, alphas, similar_items,
                                                                          similar_products_dict)
    print(best_basic, best_param_basic)
    x, y = create_x_y(all_params_basic, 1, 0)
    plot3(alphas, y, performance_measures, 'Alpha',
          'Performance of customer-customer using a meta data similarity matrix')

    # factor = 30, iteration = 60
    best, best_params, all_params = gridsearch_als(data, factors, [60], similar_items, similar_products_dict)
    x_als_factor, y_als_factor = create_x_y(all_params, 2, 0)
    plot3(x_als_factor, y_als_factor, performance_measures, 'Factors', 'Performance of ALS')

    best, best_params, all_params = gridsearch_als(data, [30], iterations, similar_items, similar_products_dict)
    x_als_iteration, y_als_iteration = create_x_y(all_params, 2, 1)
    plot3(x_als_factor, y_als_iteration, performance_measures, 'Iterations', 'Performance of ALS')

    # k = 40, 130
    best, best_params, all_params = pcf.gridsearch(data, k_s_product, similar_items, similar_products_dict)
    x_product_cf, y_product_cf = create_x_y(all_params, 1, 0)
    plot3(x_product_cf, y_product_cf, performance_measures, 'k',
          'Performance of product-product collaborative filtering')

    # k = 200
    best, best_params, all_params = ccf.gridsearch(data, k_s_customer, similar_items, similar_products_dict)
    x_customer_cf, y_customer_cf = create_x_y(all_params, 1, 0)
    plot3(x_customer_cf, y_customer_cf, performance_measures, 'k',
          'Performance of customer-customer collaborative filtering')

    # kc = 0, kp = 80, 180
    best, best_params, all_params = kunn.gridsearch(data, k_cs, [80], similar_items, similar_products_dict)
    x_kunn_k_c, y_kunn_k_c = create_x_y(all_params, 2, 0)
    plot3(x_kunn_k_c, y_kunn_k_c, performance_measures, 'k-customers', 'Performance of k-UNN')

    best, best_params, all_params = kunn.gridsearch(data, [1], k_ps, similar_items, similar_products_dict)
    x_kunn_k_p, y_kunn_k_p = create_x_y(all_params, 2, 1)
    plot3(x_kunn_k_p, y_kunn_k_p, performance_measures, 'k-products', 'Performance of k-UNN')


def gridsearch_als(data, factors, iterations, similar_items=False, similar_products_dict=None):
    best_als = [0]
    best_param_als = 0
    all_params_als = []

    for factor in factors:
        for iteration in iterations:
            model_als = ALS(factors=factor, iterations=iteration)
            performance_als = test.all_methods(model_als, data, similar_items, similar_products_dict)
            all_params_als.append([factor, iteration, performance_als])
            if performance_als[0] > best_als[0]:
                best_als = performance_als
                best_param_als = [factor, iteration]
    return best_als, best_param_als, all_params_als


# Create final deliverable
def create_recommendations(model, amount_recommendations):
    model.fit()
    recommendations_matrix = ccf.predict_recommendation(model.ratings_matrix, amount_recommendations)
    recommendations_stock_codes = np.empty(recommendations_matrix.shape, dtype=object)
    customer_map_r = list(model.customers_map)
    for stock_code, index in model.products_map.items():
        recommendations_stock_codes[recommendations_matrix == index] = str(stock_code)

    recommendations = dict()
    for i in range(len(recommendations_matrix)):
        recommendations[customer_map_r[i]] = recommendations_stock_codes[i].tolist()

    helper_functions.save_dict(recommendations, 'recommendations.json')


if __name__ == '__main__':
    main()
