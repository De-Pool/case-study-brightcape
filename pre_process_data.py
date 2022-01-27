import csv

import matplotlib.pyplot as plt
from tqdm import tqdm

import config

if config.use_cupy:
    import cupy as np
else:
    import numpy as np
import openpyxl
import pandas as pd
import seaborn as sns


def process_data(filename):
    # Use .csv since it is way faster than .xslx
    try:
        df_raw = pd.read_csv("./data/data-raw.csv", header=0, delimiter=",")
    except IOError:
        print("Didn't find a converted .csv from the .xslx file, creating it...")
        # Load the .xslx file
        xslx_file = openpyxl.load_workbook(filename).active
        # Create csv file
        csv_file = csv.writer(open("./data/data-raw.csv", 'w', newline=""))
        # Read the excel file per row and write it to the .csv file
        for row in xslx_file.rows:
            csv_file.writerow([cell.value for cell in row])
        df_raw = pd.read_csv("./data/data-raw.csv", header=0, delimiter=",")

    return df_raw


def clean_data(df, plot=False):
    if plot:
        # Inspect missing data
        for col in df.columns:
            print('{} - {}%'.format(col, round(np.mean(df[col].isnull()) * 100)))

        # Visualize it with a plot
        ax = sns.heatmap(df[df.columns].isnull())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, rotation_mode='anchor', ha='right')
        plt.tight_layout()
        plt.show()

    # Filter out the cancellations.
    # If an InvoiceNo starts with a C, it's a cancellation
    df = df.loc[~df.InvoiceNo.str.startswith('C')]

    # We only need to know whether a customer has bough a product,
    # so we filter out all quantities of 0 or lower
    df = df.loc[df.Quantity > 0]

    # We only want products with a positive price.
    df = df.loc[df.UnitPrice > 0]

    # Filter out the following Stock Codes, since they're useless
    stockcodes = ['S', 'POST', 'M', 'DOT', 'D', 'CRUK', 'C2', 'BANK CHARGES']
    df = df.loc[~df.StockCode.isin(stockcodes)]

    # We drop the observations with a missing CustomerID, as they are of no use
    df = df.dropna(axis=0, subset=["CustomerID"])

    # Drop all items which have been bought 5 times or less
    df = df.groupby('StockCode').filter(lambda x: len(x) > 5)

    # Drop all Customers which have bought 5 products or less.
    df = df.groupby('CustomerID').filter(lambda x: len(x) > 5)

    # We don't need InvoiceNo, this might be useful in more complex models
    df = df.drop(['InvoiceNo'], axis=1)

    # Drop any remaining duplicates
    df = df.drop_duplicates()

    return df


def create_customer_product_matrix(df_clean):
    # If the matrix hasn't been saved locally, compute the matrix
    # Group by CustomerID and StockCode and sum over the quantity
    # (some customers have bought a product more than once)
    df_clean = df_clean.groupby(['CustomerID', 'StockCode']).agg({'Quantity': ['sum']}).reset_index()

    # Get unique customers and unique products, n customers, m products
    unique_customers = df_clean['CustomerID'].unique()
    unique_products = df_clean['StockCode'].unique()
    n = len(unique_customers)
    m = len(unique_products)

    # Create a 1 to 1 mapping for both customers and products, 3660 -> 3112 , 4334 -> 4033
    # (map the CustomerID and StockCode to an index, ranging from (0, n) and (0, m)
    customers_map = dict()
    for i in range(n):
        customers_map[str(unique_customers[i])] = i

    products_map = dict()
    for i in range(m):
        products_map[str(unique_products[i])] = i

    # Create a n x m matrix
    matrix = np.zeros((n, m))

    for row in tqdm(df_clean.values):
        row_index = customers_map[str(row[0])]
        col_index = products_map[row[1]]
        matrix[row_index][col_index] = 1

    return matrix, customers_map, products_map
