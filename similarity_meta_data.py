import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def meta_data_similarity_matrix(df, customers_map, n):
    df = similarity_meta_data(df)
    # Create a n x m matrix
    matrix = np.zeros((n, 3))
    for row in df.values:
        index = customers_map[str(row[0])]
        matrix[index][0] = row[1]
        matrix[index][1] = row[2]
        matrix[index][2] = row[3]

    # For each customer, compute how similar they are to each other customer.
    if config.use_cupy:
        similarity_matrix = cosine_similarity(sparse.csr_matrix(np.asnumpy(matrix)))
    else:
        similarity_matrix = cosine_similarity(sparse.csr_matrix(matrix))

    return similarity_matrix


# Add columns: Country, WorkingHours, Spender to augment the similarity scores with more data.
def similarity_meta_data(df):
    # Country
    codes, uniques = pd.factorize(df['Country'])
    df['CountryCode'] = codes / (len(uniques))

    # WorkingHours
    df['WorkingHours'] = 0
    df.loc[(df.InvoiceDate.str[-8:-6].astype(int) <= 17)
           & (df.InvoiceDate.str[-8:-6].astype(int) >= 9), 'WorkingHours'] = 1

    # Spender
    df['Price'] = df['Quantity'] * df['UnitPrice']
    df = df.groupby(['CustomerID']).agg({'Price': 'sum',
                                         'CountryCode': 'max',
                                         'WorkingHours': 'max'}).reset_index()

    df['Spender'] = df.apply(spender, axis=1)
    df = df.drop(['Price'], axis=1)

    return df


def spender(df):
    # everybody below 25% percentile (305.56 dollars) = low spenders
    # everybody between 25% and 50% percentile (305.56 - 1631.6225) = medium spender
    # everybody above 75% percentile (1631.6225) = high spender
    if df['Price'] <= 305.56:
        return 0
    elif 305.56 < df['Price'] < 1631.6225:
        return 1 / 3
    elif df['Price'] > 1631.6225:
        return 2 / 3
    else:
        return 1
