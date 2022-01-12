import pandas as pd
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import csv
import json
from scipy import spatial


# The idea of this method is to solve the challenge by reducing it
# to an instance of a collaborative filtering problem, with binary, positive only data.
# There are multiple algorithms which can be used to do this, the most basic one will be using
# a cosine similarity where the most-frequent item will be recommended.
# A more complex method will be using k-Unified Nearest Neighbours (k-UNN)
# Another method which will be explored is Alternating Least Squares.
def main():
    # Process data
    filename = './data/data-raw.xlsx'
    df_raw = process_data(filename=filename)

    # Clean data
    df_clean = clean_data(df_raw)

    # Create a customer - product matrix (n x m)
    matrix, customers_map, products_map = create_customer_product_matrix(df_clean)

    # Create user - user similarity matrix (n x n)
    similarity_matrix = create_similarity_matrix(matrix, len(customers_map))

    # k nearest neighbours -> based on user similarity
    k = 25
    recommendation = predict_recommendation(matrix, similarity_matrix, len(customers_map), k)


def process_data(filename):
    # Use .csv since it is way faster than .xslx
    try:
        df_raw = pd.read_csv("./data/data-raw.csv", header=0, delimiter=",")
        return df_raw
    except:
        print("Didn't find a converted .csv from the .xslx file, creating it...")
    finally:
        # Load the .xslx file
        xslx_file = openpyxl.load_workbook(filename).active
        # Create csv file
        csv_file = csv.writer(open("./data/data-raw.csv", 'w', newline=""))
        # Read the excel file per row and write it to the .csv file
        for row in xslx_file.rows:
            csv_file.writerow([cell.value for cell in row])
        df_raw = pd.read_csv("./data/data-raw.csv", header=0, delimiter=",")

    return df_raw


def clean_data(df):
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

    # We drop the observations with a missing CustomerID, as they are of no use
    df = df.dropna(axis=0, subset=["CustomerID"])

    # We don't need: InvoiceNo, InvoiceDate, Description, UnitPrice, Country
    # these might be useful in more complex models
    drop = ["InvoiceNo", "InvoiceDate", "Description", "UnitPrice", "Country"]
    df.drop(drop, inplace=True, axis=1)

    return df


def create_customer_product_matrix(df_clean):
    try:
        matrix = genfromtxt('./data/matrix.csv', delimiter=',')

        json_file = open('./data/customers_map.json')
        customers_map = json.load(json_file)

        json_file = open('./data/products_map.json')
        products_map = json.load(json_file)
        return matrix, customers_map, products_map
    except:
        print("Didn't find a save matrix, creating it...")
    finally:
        # If the matrix hasn't been saved locally, compute the matrix
        # Group by CustomerID and StockCode and sum over the quantity
        # (some customers have bought a product more than once)
        df_clean = df_clean.groupby(['CustomerID', 'StockCode']).agg({'Quantity': ['sum']}).reset_index()

        # Get unique customers and unique products, n customers, m products
        unique_customers = df_clean['CustomerID'].unique()
        unique_products = df_clean['StockCode'].unique()
        n = len(unique_customers)
        m = len(unique_products)

        # Create a 1 to 1 mapping for both customers and products
        # (map the CustomerID and StockCode to an index, ranging from (0, n) and (0, m)
        customers_map = dict()
        for i in range(n):
            customers_map[unique_customers[i]] = i
        save_dict(customers_map, 'customers_map.json')
        products_map = dict()
        for i in range(m):
            products_map[unique_products[i]] = i

        # Create a n x m matrix
        matrix = np.zeros((n, m))
        for _, row in df_clean.iterrows():
            row_index = customers_map[row['CustomerID'].values[0]]
            col_index = products_map[row['StockCode'].values[0]]
            if row['Quantity'].values[0] > 0:
                matrix[row_index][col_index] = 1

        save_dict(customers_map, 'customers_map.json')
        save_dict(products_map, 'products_map.json')
        save_matrix(matrix, 'matrix.csv')

        return matrix, customers_map, products_map


def create_similarity_matrix(c_p_matrix, n):
    try:
        similarity_matrix = genfromtxt('./data/similarity_matrix.csv', delimiter=',')
        return similarity_matrix
    except:
        print("Didn't find a similarity matrix, creating it...")
    finally:
        similarity_matrix = np.zeros((n, n))

        # For each user, compute how similar they are to each other user.
        for i in range(n):
            for j in range(n):
                # cosine similarity = 1 - cosine distance
                similarity_matrix[i][j] = 1 - spatial.distance.cosine(c_p_matrix[i], c_p_matrix[j])

        save_matrix(similarity_matrix, 'similarity_matrix.csv')

        return similarity_matrix


def predict_recommendation(c_p_matrix, similarity_matrix, n, k):
    try:
        recommendations = genfromtxt('./data/recommendations.csv', delimiter=',')
        return recommendations
    except:
        print("Didn't find recommendations, creating it...")
    finally:
        recommendations = np.zeros((n, k))
        for i in range(n):
            pass
        save_matrix(recommendations, 'recommendations.csv')

        return recommendations


def save_dict(dictionary, name):
    json_file = open('./data/' + name, 'w')
    json.dump(dictionary, json_file)


def save_matrix(matrix, name):
    np.savetxt('./data/' + name, matrix, delimiter=',')


if __name__ == '__main__':
    main()
