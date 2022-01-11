import pandas as pd
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import csv


def main():
    # Process data
    filename = './data/data-raw.xlsx'
    df_raw = process_data(filename=filename, savedcsv=True)

    # Clean data
    df_clean = clean_data(df_raw)

    # Create a customer - product matrix (n x m)
    saved_csv = False
    matrix, customers_map, products_map = create_matrix(df_clean, savedcsv=saved_csv)
    if not saved_csv:
        # Save the intermediary result
        save_matrix(matrix, customers_map, products_map)


def process_data(filename, savedcsv):
    # Use .csv since it is way faster than .xslx
    if not savedcsv:
        # Load the .xslx file
        xslx_file = openpyxl.load_workbook(filename).active
        # Create csv file
        csv_file = csv.writer(open("./data/data-raw.csv",
                                   'w',
                                   newline=""))
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

    # We drop the observations with a missing CustomerID, as they are of no use
    df = df.dropna(axis=0, subset=["CustomerID"])

    # We don't need: InvoiceNo, InvoiceDate, Description, UnitPrice, Country
    # these might be useful in more complex models
    drop = ["InvoiceNo", "InvoiceDate", "Description", "UnitPrice", "Country"]
    df.drop(drop, inplace=True, axis=1)

    return df


def create_matrix(df_clean, savedcsv):
    if not savedcsv:
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
    else:
        matrix = genfromtxt('./data/matrix.csv', delimiter=',')

        csv_file = open('./data/customers_map.csv', 'w')
        reader = csv.reader(csv_file)
        customers_map = dict(reader)

        csv_file = open('./data/products_map.csv', 'w')
        reader = csv.reader(csv_file)
        products_map = dict(reader)

    return matrix, customers_map, products_map


def save_matrix(matrix, customers_map, products_map):
    np.savetxt('./data/matrix.csv', matrix, delimiter=',')

    csv_file = open('./data/customers_map.csv', 'w')
    writer = csv.writer(csv_file)
    for key, value in customers_map.items():
        writer.writerow([key, value])

    csv_file = open('./data/products_map.csv', 'w')
    writer = csv.writer(csv_file)
    for key, value in products_map.items():
        writer.writerow([key, value])


if __name__ == '__main__':
    main()
