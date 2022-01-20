import math

import numpy as np
from implicit.approximate_als import AlternatingLeastSquares as ALS
from implicit.cpu.bpr import BayesianPersonalizedRanking as BPR
from implicit.lmf import LogisticMatrixFactorization as LMF
from scipy import sparse

import basic_collaborative_filtering as bcf


def compute_performance_single(recommendations, test_data, n, r, decimals):
    total_hits = 0
    total_ndcg = 0
    total_map = 0

    for i in range(len(test_data)):
        if int(test_data[i]) in recommendations[i].astype(int):
            # indexing starts at 0, so we add 1 to get a correct rank.
            rank = np.where(recommendations[i].astype(int) == int(test_data[i]))[0] + 1
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


def all_methods(model, r, decimals=4, similar_items=False):
    if isinstance(model, (ALS, BPR, LMF)):
        # r = [train_matrix, split_method, r, products_map, df_clean]
        matrix = sparse.csr_matrix(r['train_matrix'])
        model.fit(matrix.T)
        recommendations = np.zeros((len(r['train_matrix']), r['r']))
        for i in range(len(r['train_matrix'])):
            recommendations[i] = np.array(list(dict(model.recommend(i, matrix, N=r['r']))))

        if similar_items:
            extra_recommendations = find_similar_items(r['test_data'], recommendations, r['products_map'],
                                                       r['df_clean'])

            new_test_set = dict()
            for i in range(len(r['test_data'])):
                if i in extra_recommendations:
                    test_instances = [*[r['test_data'][i]], *extra_recommendations[i]]
                else:
                    test_instances = [r['test_data'][i]]
                new_test_set[i] = test_instances
            return compute_performance_multiple(recommendations, new_test_set, len(r['train_matrix']), decimals)
        elif isinstance(r['test_data'], dict):
            return compute_performance_multiple(recommendations, r['test_data'], len(recommendations), decimals)
        else:
            return compute_performance_single(recommendations, r['test_data'], len(r['test_data']),
                                              len(recommendations), decimals)
    else:
        recommendations = bcf.predict_recommendation(model.ratings_matrix, r)
        if similar_items:
            extra_recommendations = find_similar_items(model.test_data, recommendations, model.products_map,
                                                       model.df_clean)

            new_test_set = dict()
            for i in range(model.n):
                #TODO fix
                if i in extra_recommendations:
                    test_instances = [*[model.test_data[i]], *extra_recommendations[i]]
                else:
                    test_instances = [model.test_data[i]]
                new_test_set[i] = test_instances
            return compute_performance_multiple(recommendations, new_test_set, model.n, decimals)
        elif isinstance(model.test_data, dict):
            return compute_performance_multiple(recommendations, model.test_data, model.n, decimals)
        else:
            return compute_performance_single(recommendations, model.test_data, model.n, r, decimals)


# test_set is either an array or dict, recommendations is matrix
# returns a dict, which is the new test set
def find_similar_items(test_set, recommendations, products_map, df_clean):
    stock_codes_df = df_clean[['StockCode', 'Description', 'UnitPrice']] \
        .drop_duplicates(subset=['StockCode'])
    stock_codes_df = stock_codes_df.astype({'StockCode': 'string', 'UnitPrice': 'float', 'Description': 'string'})

    if isinstance(test_set, dict):
        pass
        # recommendations_stock_codes = np.empty(recommendations.shape, dtype="S6")
        # test_set_stock_codes = dict()
        # for stock_code, test_instances in test_set.items():
        #     test_set_stock_codes
        #
        # for stock_code, index in products_map.items():
        #     recommendations_stock_codes[recommendations == index] = str(stock_code)
        #
        #
        # correct_recommendations = dict()
        # for i in range(len(test_set_stock_codes)):
        #     similar_items = filter_similar_items_single(test_set_stock_codes[i], recommendations_stock_codes[i],
        #                                                 stock_codes_df, products_map)
        #     if len(similar_items) > 0:
        #         correct_recommendations[i] = similar_items
        #
        # return correct_recommendations
    else:
        recommendations_stock_codes = np.empty(recommendations.shape, dtype="S6")
        test_set_stock_codes = np.empty(test_set.shape, dtype="S6")
        for stock_code, index in products_map.items():
            recommendations_stock_codes[recommendations == index] = str(stock_code)
            test_set_stock_codes[test_set == index] = str(stock_code)

        correct_recommendations = dict()
        for i in range(len(test_set_stock_codes)):
            similar_items = filter_similar_items_single(test_set_stock_codes[i], recommendations_stock_codes[i],
                                                        stock_codes_df, products_map)
            if len(similar_items) > 0:
                correct_recommendations[i] = similar_items

        return correct_recommendations


def filter_similar_items_single(test_stock_code, recommendations_stock_codes, stock_code_df, products_map):
    similar_items = []
    test_instance = stock_code_df[stock_code_df.StockCode == str(test_stock_code, 'UTF-8')]
    if test_instance.empty:
        return []
    upper_price = test_instance.UnitPrice.values[0] * 1.1
    lower_price = test_instance.UnitPrice.values[0] * 0.9
    test_str_set = set(test_instance.Description.values[0].split(" "))
    for stock_code in recommendations_stock_codes:
        if stock_code[:3] != test_stock_code[:3] or stock_code == test_stock_code:
            continue
        recommendation = stock_code_df[stock_code_df.StockCode == str(stock_code, 'UTF-8')]
        if recommendation.empty or recommendation.UnitPrice.values[0] > upper_price or recommendation.UnitPrice.values[0] < lower_price:
            continue
        rec_str_set = set(recommendation.Description.values[0].split(" "))
        if len(test_str_set.intersection(rec_str_set)) >= 1:
            similar_items.append(products_map[str(stock_code, 'UTF-8')])
    return similar_items


def filter_similar_items_multiple(test_stock_codes, recommendations_stock_codes, stock_code_df, products_map):
    similar_items = []
    test_instances = stock_code_df[stock_code_df.StockCode.isin(test_stock_codes.astype(str))]

    # if test_instance.empty:
    #     return []
    # upper_price = test_instance.UnitPrice.values[0] * 1.1
    # lower_price = test_instance.UnitPrice.values[0] * 0.9
    # test_str_set = set(test_instance.Description.values[0].split(" "))
    # for stock_code in recommendations_stock_codes:
    #     if stock_code[:3] != test_stock_code[:3] or stock_code == test_stock_code:
    #         continue
    #     recommendation = stock_code_df[stock_code_df.StockCode == str(stock_code, 'UTF-8')]
    #     if recommendation.UnitPrice.values[0] > upper_price or recommendation.UnitPrice.values[0] < lower_price:
    #         continue
    #     rec_str_set = set(recommendation.Description.values[0].split(" "))
    #     if len(test_str_set.intersection(rec_str_set)) >= 1:
    #         similar_items.append(products_map[str(stock_code, 'UTF-8')])
    return similar_items
