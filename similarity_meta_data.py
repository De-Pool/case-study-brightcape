import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
import pandas as pd
from scipy import spatial

import helper_functions as hf


def meta_data_similarity_matrix(df, customers_map, n):
    try:
        similarity_matrix = hf.read_matrix('meta_data/similarity_matrix.csv')
    except IOError:
        print("Didn't find a meta data similarity matrix, creating it...")
        df = similarity_meta_data(df)
        # Create a n x m matrix
        matrix = np.zeros((n, 3))
        for _, row in df.iterrows():
            index = customers_map[str(row['CustomerID'])]
            matrix[index][0] = row['CountryCode']
            matrix[index][1] = row['WorkingHours']
            matrix[index][2] = row['Spender']

        similarity_matrix = np.zeros((n, n))
        # For each customer, compute how similar they are to each other customer.
        for i in range(n):
            for j in range(n):
                pass
                # cosine similarity = 1 - cosine distance
                similarity_matrix[i][j] = 1 - spatial.distance.cosine(matrix[i], matrix[j])

        hf.save_matrix(similarity_matrix, 'meta_data/similarity_matrix.csv')

    return similarity_matrix


# Add columns: Country, WorkingHours, Spender to augment the similarity scores with more data.
def similarity_meta_data(df):
    # Country
    codes, uniques = pd.factorize(df['Country'])
    df['CountryCode'] = codes

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
    df.drop('Price', inplace=True, axis=1)

    return df


def spender(df):
    # everybody below 25% percentile (305.56 dollars) = low spenders
    # everybody between 25% and 50% percentile (305.56 - 1631.6225) = medium spender
    # everybody above 75% percentile (1631.6225) = high spender
    if df['Price'] <= 305.56:
        return 0
    elif 305.56 < df['Price'] < 1631.6225:
        return 1
    elif df['Price'] > 1631.6225:
        return 2
    else:
        return 3
