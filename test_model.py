import math

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from implicit.approximate_als import AlternatingLeastSquares as ALS
from implicit.cpu.bpr import BayesianPersonalizedRanking as BPR
from implicit.lmf import LogisticMatrixFactorization as LMF
from implicit.nearest_neighbours import BM25Recommender as BM25
from implicit.nearest_neighbours import TFIDFRecommender as TFIDF
from scipy import sparse

import models.customer_cf as ccf
import helper_functions as hf


def compute_performance_single(recommendations, test_data, n, r, decimals):
    total_hits = 0
    total_ndcg = 0
    total_map = 0

    for i in range(len(test_data)):
        if test_data[i] in recommendations[i].astype(int):
            # indexing starts at 0, so we add 1 to get a correct rank.
            rank = np.where(recommendations[i].astype(int) == test_data[i])[0] + 1
            total_hits += 1
            total_ndcg += math.log(2) / math.log(rank + 1)
            total_map += 1 / r

    ndcg = round(total_ndcg / n, decimals)
    hit_rate = round(total_hits / n, decimals)
    map = round(total_map / n, decimals)

    return hit_rate, ndcg, map


def compute_performance_multiple(recommendations, test_data, n, decimals):
    total_average_precision = 0
    total_hits = 0
    total = 0
    dcg = 0
    idcg = 0
    for customerID, test_instances in test_data.items():
        total += len(test_instances)
        hits = np.intersect1d(np.array(test_instances), np.array(recommendations[customerID]))
        total_hits += len(hits)
        if len(hits) == 0:
            continue

        relevant_products = 0
        total_precision = 0
        # Calculate DCG and IDCG
        for j in range(len(hits)):
            # indexing starts at 0, so we add 1 to get a correct rank.
            rank = np.where(recommendations[customerID] == int(hits[j]))[0][0] + 1

            # Precision is correct positives / predicted positives
            relevant_products += 1
            total_precision += relevant_products / rank

            dcg += 1 / math.log(rank + 1)
            idcg += 1 / math.log(j + 2)
        total_average_precision += total_precision / relevant_products

    # Calculate hit_rate and ndcg
    hit_rate = round(total_hits / total, decimals)
    ndcg = round(dcg / idcg, decimals)
    map = round(total_average_precision / n, decimals)

    return hit_rate, ndcg, map


def all_methods(model, r, similar_items, similar_products_dict, decimals=4):
    if isinstance(model, (ALS, BPR, LMF, BM25, TFIDF)):
        matrix = sparse.csr_matrix(r['train_matrix'])
        model.fit(matrix.T, show_progress=False)

        recommendations = np.zeros((len(r['train_matrix']), r['r']))
        for i in range(len(r['train_matrix'])):
            recommendation = np.array(list(dict(model.recommend(i, matrix, N=r['r']))))
            recommendations[i, :len(recommendation)] = recommendation
        test_data = r['test_data']
        df_clean = r['df_clean']
        products_map = r['products_map']
        n = r['n']
        r = r['r']
    else:
        recommendations = ccf.predict_recommendation(model.ratings_matrix, r)
        test_data = model.test_data
        df_clean = model.df_clean
        products_map = model.products_map
        n = model.n

    if similar_items:
        extra_recommendations = find_extra_recommendations(test_data, recommendations, products_map, similar_products_dict)

        new_test_set = dict()
        for i in range(n):
            new_test_set[i] = hf.concat(test_data[i], extra_recommendations[i])

        return compute_performance_multiple(recommendations, new_test_set, n, decimals)
    elif isinstance(test_data, dict):
        return compute_performance_multiple(recommendations, test_data, n, decimals)
    else:
        return compute_performance_single(recommendations, test_data, n, r, decimals)


# test_set is either an array or dict, recommendations is matrix
# returns a dict, which is the new test set
def find_extra_recommendations(test_set, recommendations, products_map, similar_products_dict):
    products_map_r = list(products_map)
    recommendations_stock_codes = np.empty(recommendations.shape, dtype=object)
    test_set_stock_codes = dict()
    extra_recommendations = dict()

    for stock_code, index in products_map.items():
        recommendations_stock_codes[recommendations == index] = str(stock_code)

    if isinstance(test_set, dict):
        for customerID, test_instances in test_set.items():
            test_set_stock_codes[customerID] = list(map(lambda x: products_map_r[x], test_instances))

        for customerID, test_instances in test_set_stock_codes.items():
            similar_products = []
            for stock_code in test_instances:
                common_products = set(similar_products_dict[stock_code]).intersection(
                    set(recommendations_stock_codes[customerID]))
                similar_products = hf.concat(similar_products, list(common_products))
            extra_recommendations[customerID] = similar_products
    else:
        test_set_stock_codes = np.empty(test_set.shape, dtype=object)
        for stock_code, index in products_map.items():
            test_set_stock_codes[test_set == index] = str(stock_code)

        for i in range(len(test_set_stock_codes)):
            extra_recommendations[i] = list(
                set(similar_products_dict[test_set_stock_codes[i]]).intersection(set(recommendations_stock_codes[i])))

    for customerID, test_instances in extra_recommendations.items():
        extra_recommendations[customerID] = list(map(lambda x: products_map[x], test_instances))

    return extra_recommendations


def compute_similar_products(df_clean):
    try:
        similar_products_dict = hf.read_dict('similar_products_dict.json')
    except IOError:
        stock_codes_df = df_clean[['StockCode', 'Description', 'UnitPrice']].drop_duplicates(subset=['StockCode'])
        stock_codes_df = stock_codes_df.astype({'StockCode': 'string', 'UnitPrice': 'float', 'Description': 'string'})
        stock_codes_df = stock_codes_df.set_index('StockCode', drop=False)

        # 3 heuristics: the first 3 characters of the stockcode match
        # the unitprice differs by at most 10%
        # the description has at least 1 matching word
        def filter(stock_code, unit_price_lower, unit_price_upper, description_set):
            similar_products = stock_codes_df.apply(lambda row: (
                    (row['StockCode'][:3] == stock_code[:3] and row['StockCode'] != stock_code) and
                    (unit_price_lower <= row['UnitPrice'] <= unit_price_upper) and
                    (len(set(row['Description'].split(" ")).intersection(description_set)) >= 1)), axis=1)
            return similar_products[similar_products].index.tolist()

        similar_products_dict = stock_codes_df.apply(
            lambda row: filter(row['StockCode'], row['UnitPrice'] * 0.9, row['UnitPrice'] * 1.1,
                               set(row['Description'].split(' '))), axis=1).to_dict()
        hf.save_dict(similar_products_dict, 'similar_products_dict.json')

    return similar_products_dict
