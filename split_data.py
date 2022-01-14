import numpy as np


def leave_one_out(utility_matrix, n):
    test = []
    # For each customer, select 1 random
    for i in range(n):
        # Get all bought products from customer i
        bought_products = np.nonzero(utility_matrix[i])[0]
        if len(bought_products) > 0:
            # Choose a random product p
            p = np.random.choice(bought_products)
            test.append(p)
            utility_matrix[i][p] = 0
    return utility_matrix, test


def leave_last_out(utility_matrix, indices, n):
    for i in range(n):
        utility_matrix[i][int(indices[i])] = 0
    return utility_matrix, indices


def get_indices_last(df_clean, customers_map, products_map):
    test_set = np.zeros(len(customers_map))
    # Sort dataframe by date
    df_clean = df_clean.sort_values(['InvoiceDate'], ascending=True)
    df_clean = df_clean.groupby(['CustomerID'])
    for customerID, group in df_clean:
        last_stock_code = group['StockCode'].iloc[-1]
        test_set[customers_map[str(customerID)]] = products_map[str(last_stock_code)]
    return test_set
