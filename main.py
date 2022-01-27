import matplotlib.pyplot as plt

import config
import split_data

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from implicit.approximate_als import AlternatingLeastSquares as ALS
from implicit.nearest_neighbours import BM25Recommender as BM25
from implicit.nearest_neighbours import TFIDFRecommender as TFIDF

import models.product_cf as pcf
import models.customer_cf as ccf
import models.kunn_cf as kunn
import models.kunn_custom_cf as kunn_custom
import test_model


def train_test_size():
    # alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35,
              0.4]
    similar_items = True
    decimals = 4

    y_hit_rate = []
    y_ncdg = []
    y_map = []
    recommendations = 250

    for alpha in alphas:
        data = split_data.create_model_data('./data/data-raw.xlsx', False, split_method='temporal', alpha=alpha,
                                            r=recommendations)
        similar_products_dict = test_model.compute_similar_products(data['df_clean'])

        model_basic_cf = ccf.CollaborativeFilteringBasic('', data, k=200)
        model_basic_cf.fit()

        model_product_cf = pcf.CollaborativeFilteringProduct('', data, k=40)
        model_product_cf.fit()

        model_kunn = kunn.CollaborativeFilteringKUNN('', data, k_products=80, k_customers=0)
        model_kunn.fit()

        performance_basic = test_model.all_methods(model_basic_cf, recommendations, decimals, similar_items,
                                                   similar_products_dict)
        performance_product_cf = test_model.all_methods(model_product_cf, recommendations, decimals, similar_items,
                                                        similar_products_dict)
        performance_kunn = test_model.all_methods(model_kunn, recommendations, decimals, similar_items,
                                                  similar_products_dict)
        performance_ALS = test_model.all_methods(ALS(factors=30, iterations=50), data, decimals, similar_items,
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


def recommendations():
    rs = np.arange(1, 20, 5)

    data = split_data.create_model_data('./data/data-raw.xlsx', False, split_method='temporal', alpha=0.1)
    similar_products_dict = test_model.compute_similar_products(data['df_clean'])
    decimals = 4
    similar_items = True

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
        performance_basic = test_model.all_methods(model_basic_cf, r, decimals, similar_items, similar_products_dict)
        performance_product_cf = test_model.all_methods(model_product_cf, r, decimals, similar_items, similar_products_dict)
        performance_kunn = test_model.all_methods(model_kunn, r, decimals, similar_items, similar_products_dict)
        performance_ALS = test_model.all_methods(ALS(factors=30, iterations=50), data, decimals, similar_items, similar_products_dict)
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


# We solve the challenge by reducing it
# to an instance of a collaborative filtering problem, with binary, positive only data.
# There are multiple algorithms which can be used to do this, the most basic one will be using
# cosine similarity where the most-frequent product will be recommended.
# A more complex method will be using k-Unified Nearest Neighbours (k-UNN)
# Another method which will be explored is Alternating Least Squares.
def best_models():
    filename_xlsx = './data/data-raw.xlsx'
    # split_method: temporal, last_out, one_out
    split_method = 'ztemporal'
    data = split_data.create_model_data(filename_xlsx, False, split_method, alpha=0.1)
    similar_products_dict = test_model.compute_similar_products(data['df_clean'])
    decimals = 4
    # Get r recommendations
    recommendations = 250
    similar_items = True

    k_c_b = 200
    model_basic_cf = ccf.CollaborativeFilteringBasic(filename_xlsx, data, k_c_b)
    model_basic_cf.fit()
    performance_basic = test_model.all_methods(model_basic_cf, recommendations, decimals, similar_items,
                                               similar_products_dict)

    k_p_b = 40
    model_product_cf = pcf.CollaborativeFilteringProduct(filename_xlsx, data, k_p_b)
    model_product_cf.fit()
    performance_product_cf = test_model.all_methods(model_product_cf, recommendations, decimals, similar_items,
                                                    similar_products_dict)

    k_c_custom = 180
    k_p_custom = 10
    model_kunn_custom = kunn_custom.CollaborativeFilteringKUNNCustom(filename_xlsx, data, k_c_custom, k_p_custom)
    model_kunn_custom.fit()
    performance_kunn_custom = test_model.all_methods(model_kunn_custom, recommendations, decimals, similar_items,
                                                     similar_products_dict)

    # k nearest neighbours for customers
    k_c = 0
    # k_p nearest neighbours for products
    k_p = 80
    model_kunn = kunn.CollaborativeFilteringKUNN(filename_xlsx, data, k_p, k_c)
    model_kunn.fit()
    performance_kunn = test_model.all_methods(model_kunn, recommendations, decimals, similar_items,
                                              similar_products_dict)

    factors = 30
    iterations = 50
    performance_ALS = test_model.all_methods(ALS(factors=factors, iterations=iterations), data, decimals, similar_items,
                                             similar_products_dict)
    performance_bm25 = test_model.all_methods(BM25(), data, decimals, similar_items, similar_products_dict)
    performance_tfidf = test_model.all_methods(TFIDF(), data, decimals, similar_items, similar_products_dict)

    print("basic customer", performance_basic)
    print("basic product", performance_product_cf)
    print("kunn", performance_kunn)
    print("kunn custom", performance_kunn_custom)
    print("als", performance_ALS)
    print("bm25", performance_bm25)
    print("tfidf", performance_tfidf)


def grid_search():
    # best_models()
    recommendations = 250
    split_method = 'temporal'
    data = split_data.create_model_data('./data/data-raw.xlsx', False, split_method, alpha=0.1, r=recommendations)
    similar_products_dict = test_model.compute_similar_products(data['df_clean'])

    similar_items = False
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

    # k_c = 140, k_p = 180
    best, best_params, all_params = gridsearch_kunn_custom(data, k_cs, [180], similar_items, similar_products_dict)
    print(best, best_params)
    x_customer_kunn_c, y_customer_kunn_c = create_x_y(all_params, 2, 0)
    plot3(x_customer_kunn_c, y_customer_kunn_c, performance_measures, 'k-customer',
          'Performance of k-UNN custom collaborative filtering')

    best, best_params, all_params = gridsearch_kunn_custom(data, [10], k_ps, similar_items, similar_products_dict)
    print(best, best_params)
    x_product_kunn_c, y_product_kunn_c = create_x_y(all_params, 2, 1)
    plot3(x_product_kunn_c, y_product_kunn_c, performance_measures, 'k-product',
          'Performance of k-UNN custom collaborative filtering')

    # factor = 30, iteration = 60
    best, best_params, all_params = gridsearch_als(data, factors, [60], similar_items, similar_products_dict)
    print(best, best_params)
    x_als_factor, y_als_factor = create_x_y(all_params, 2, 0)
    plot3(x_als_factor, y_als_factor, performance_measures, 'Factors', 'Performance of ALS')

    best, best_params, all_params = gridsearch_als(data, [30], iterations, similar_items, similar_products_dict)
    print(best, best_params)
    x_als_iteration, y_als_iteration = create_x_y(all_params, 2, 1)
    plot3(x_als_factor, y_als_iteration, performance_measures, 'Iterations', 'Performance of ALS')

    # k = 40, 130
    best, best_params, all_params = gridsearch_product(data, k_s_product, similar_items, similar_products_dict)
    print(best, best_params)
    x_product_cf, y_product_cf = create_x_y(all_params, 1, 0)
    plot3(x_product_cf, y_product_cf, performance_measures, 'k',
          'Performance of product-product collaborative filtering')

    # k = 200
    best, best_params, all_params = gridsearch_customer(data, k_s_customer, similar_items, similar_products_dict)
    print(best, best_params)
    x_customer_cf, y_customer_cf = create_x_y(all_params, 1, 0)
    plot3(x_customer_cf, y_customer_cf, performance_measures, 'k',
          'Performance of customer-customer collaborative filtering')

    # kc = 0, kp = 80, 180
    best, best_params, all_params = gridsearch_kunn(data, k_cs, [80], similar_items, similar_products_dict)
    print(best, best_params)
    x_kunn_k_c, y_kunn_k_c = create_x_y(all_params, 2, 0)
    plot3(x_kunn_k_c, y_kunn_k_c, performance_measures, 'k-customers', 'Performance of k-UNN')

    best, best_params, all_params = gridsearch_kunn(data, [1], k_ps, similar_items, similar_products_dict)
    print(best, best_params)
    x_kunn_k_p, y_kunn_k_p = create_x_y(all_params, 2, 1)
    plot3(x_kunn_k_p, y_kunn_k_p, performance_measures, 'k-products', 'Performance of k-UNN')


def gridsearch_als(data, factors, iterations, similar_items=False, similar_products_dict=None):
    best_als = [0]
    best_param_als = 0
    all_params_als = []

    for factor in factors:
        for iteration in iterations:
            model_als = ALS(factors=factor, iterations=iteration)
            performance_als = test_model.all_methods(model_als, data, similar_items, similar_products_dict)
            all_params_als.append([factor, iteration, performance_als])
            if performance_als[0] > best_als[0]:
                best_als = performance_als
                best_param_als = [factor, iteration]
    return best_als, best_param_als, all_params_als


def gridsearch_product(data, k_s, similar_items=False, similar_products_dict=None):
    best_basic = [0]
    best_param_basic = 0
    all_params_basic = []
    model_product_cf = pcf.CollaborativeFilteringProduct('', data, 1, 0, False, False)
    model_product_cf.fit()
    for k in k_s:
        model_product_cf.k = k
        model_product_cf.predict_ratings_matrix()
        performance_basic = test_model.all_methods(model_product_cf, data['r'], similar_items, similar_products_dict)
        all_params_basic.append([k, performance_basic])
        if performance_basic[0] > best_basic[0]:
            best_basic = performance_basic
            best_param_basic = k
    return best_basic, best_param_basic, all_params_basic


def gridsearch_customer(data, k_s, similar_items=False, similar_products_dict=None):
    best_basic = [0]
    best_param_basic = 0
    all_params_basic = []
    model_basic_cf = ccf.CollaborativeFilteringBasic('', data, 1, 0, False, False)
    model_basic_cf.fit()
    for k in k_s:
        model_basic_cf.k = k
        model_basic_cf.predict_ratings_matrix_fast()
        performance_basic = test_model.all_methods(model_basic_cf, data['r'], similar_items, similar_products_dict)
        all_params_basic.append([k, performance_basic])
        if performance_basic[0] > best_basic[0]:
            best_basic = performance_basic
            best_param_basic = k
    return best_basic, best_param_basic, all_params_basic


def gridsearch_kunn(data, k_cs, k_ps, similar_items=False, similar_products_dict=None):
    best_kunn = [0]
    best_params_kunn = []
    all_params_kunn = []

    model_kunn = kunn.CollaborativeFilteringKUNN('', data, 1, 1, 0, False, False)
    model_kunn.create_similarity_matrices_fast()
    for k_c in k_cs:
        for k_p in k_ps:
            model_kunn.k_products = k_p
            model_kunn.k_customers = k_c

            model_kunn.predict_ratings_matrix_fast(True)
            performance_kunn = test_model.all_methods(model_kunn, data['r'], similar_items, similar_products_dict)

            all_params_kunn.append([k_c, k_p, performance_kunn])
            if performance_kunn[0] > best_kunn[0]:
                best_kunn = performance_kunn
                best_params_kunn = [k_c, k_p]

    return best_kunn, best_params_kunn, all_params_kunn


def gridsearch_kunn_custom(data, k_cs, k_ps, similar_items=False, similar_products_dict=None):
    best_kunn_custom = [0]
    best_params_kunn_custom = []
    all_params_kunn_custom = []

    model_kunn_custom = kunn_custom.CollaborativeFilteringKUNNCustom('', data, 1, 1, 0, False, False)
    model_kunn_custom.create_similarity_matrices()
    for k_c in k_cs:
        for k_p in k_ps:
            model_kunn_custom.k_p = k_p
            model_kunn_custom.k_c = k_c

            model_kunn_custom.predict_ratings_matrix(True)
            performance_kunn_custom = test_model.all_methods(model_kunn_custom, data['r'], similar_items,
                                                             similar_products_dict)

            all_params_kunn_custom.append([k_c, k_p, performance_kunn_custom])
            if performance_kunn_custom[0] > best_kunn_custom[0]:
                best_kunn_custom = performance_kunn_custom
                best_params_kunn_custom = [k_c, k_p]

    return best_kunn_custom, best_params_kunn_custom, all_params_kunn_custom


def create_x_y(all_params, num_params, parameter):
    x = []
    y = []
    for i in range(len(all_params)):
        x.append(all_params[i][parameter])
        y.append([all_params[i][num_params][0], all_params[i][num_params][1], all_params[i][num_params][2]])
    return np.array(x), np.array(y).T


def plot3(xs, ys, labels, x_lbl, title, legend=True):
    fig, host = plt.subplots(figsize=(8, 5))

    par1 = host.twinx()
    par2 = host.twinx()

    host.set_xlabel(x_lbl)
    host.set_ylabel(labels[0])
    par1.set_ylabel(labels[1])
    par2.set_ylabel(labels[2])

    line1, = host.plot(xs, ys[0], color='red', label=labels[0])
    line2, = par1.plot(xs, ys[1], color='green', label=labels[1])
    line3, = par2.plot(xs, ys[2], color='blue', label=labels[2])

    if legend:
        host.legend(handles=[line1, line2, line3], loc='lower right')

    par2.spines['right'].set_position(('outward', 60))
    host.yaxis.label.set_color(line1.get_color())
    par1.yaxis.label.set_color(line2.get_color())
    par2.yaxis.label.set_color(line3.get_color())

    plt.title(title)
    fig.tight_layout()
    plt.show()


def plot2(xs, ys, labels, x_lbl, y_lbls, title, legend=True):
    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    ax.plot(xs[0], ys[0], label=labels[0], color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbls[0])

    # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()
    ax2.plot(xs[1], ys[1], label=labels[1], color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylabel(y_lbls[1])
    plt.title(title)
    if legend:
        fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.14))
        plt.tight_layout()
    plt.show()


def plot(x, y, label, x_lbl, y_lbl, title, legend=True):
    for i in range(len(y)):
        plt.plot(x, y[i], label=label[i])
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    if legend:
        plt.legend()
    plt.show()


if __name__ == '__main__':
    best_models()
    train_test_size()
    recommendations()
    grid_search()
