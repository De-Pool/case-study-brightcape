import copy

import pandas as pd

import config
import pre_process_data as ppd

if config.use_cupy:
    import cupy as np
else:
    import numpy as np


def temporal_split(df_clean, alpha=0.05):
    # Sort dataframe by date
    df_clean = df_clean.sort_values(['InvoiceDate'], ascending=True)

    # Generate a test and train set
    f = lambda g: g.iloc[len(g) - 1:] \
        if (1 / len(g)) > alpha \
        else g.iloc[int(round(len(g) * (1 - alpha), 0)):]
    test = df_clean.groupby('CustomerID').apply(f).reset_index(drop=True)
    train = pd.concat([df_clean, test]).drop_duplicates(keep=False)
    df_clean = pd.concat([train, test])

    # Generate a matrix, customers_map, products_map based on the remaining data
    matrix, customers_map, products_map = ppd.create_customer_product_matrix(df_clean)

    # Create a test_set, which is a dict, with as key customerID and as value a list of product indices
    test_set = dict()
    test = test[['CustomerID', 'StockCode']].drop_duplicates().groupby('CustomerID')
    for customerID, group in test:
        arr_map = list(map(lambda x: int(products_map[x]), group.StockCode.values))
        test_set[customers_map[str(customerID)]] = arr_map

    # For each test instance, set the value to 0 in the train matrix.
    train_matrix = copy.deepcopy(matrix)
    for customerID, stockCodes in test_set.items():
        for stockCode in stockCodes:
            train_matrix[customerID][stockCode] = 0

    return matrix, customers_map, products_map, train_matrix, test_set, df_clean


def leave_one_out(utility_matrix, n):
    # Index i in the test set is the index of customer u
    test_set = np.zeros(n, dtype=int)
    # For each customer, select 1 random
    for i in range(n):
        # Get all bought products from customer i
        bought_products = np.nonzero(utility_matrix[i])[0]
        if len(bought_products) > 0:
            # Choose a random product p
            p = np.random.choice(bought_products)
            test_set[i] = p
            utility_matrix[i][p] = 0
    return utility_matrix, test_set


def leave_last_out(utility_matrix, df_clean, customers_map, products_map):
    indices = get_indices_last(df_clean, customers_map, products_map)
    # Index i in the test set is the index of customer u
    for i in range(len(customers_map)):
        utility_matrix[i][int(indices[i])] = 0
    return utility_matrix, indices


def get_indices_last(df_clean, customers_map, products_map):
    test_set = np.zeros(len(customers_map), dtype=int)
    # Sort dataframe by date
    df_clean = df_clean.sort_values(['InvoiceDate'], ascending=True)
    df_clean = df_clean.groupby(['CustomerID'])
    for customerID, group in df_clean:
        last_stock_code = group['StockCode'].iloc[-1]
        test_set[customers_map[str(customerID)]] = int(products_map[str(last_stock_code)])
    return test_set
