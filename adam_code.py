import pandas as pd
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import csv
import json
from scipy import spatial
import collections

def main():
    # Process data
    filename = './data/data-raw.xlsx'
    df_raw = process_data(filename=filename)

    # Clean data
    df_clean = clean_data(df_raw, False)
    df_with_price = add_price(df_clean)
    df_collapsed = collapse_df(df_with_price)
    print('worked')
    print(df_collapsed.shape[0])
    print(df_collapsed.describe())
    

    print('')
    for index in range(df_collapsed.shape[0]):
        if df_collapsed.loc[index, 'Price'].values[0] < 1:
            print('errorr')
            print(df_collapsed.loc[index, 'CustomerID'])

    print('')
    money_distribution(df_collapsed, False)

def process_data(filename):
    # Use .csv since it is way faster than .xslx
    try:
        df_raw = pd.read_csv("./data/data-raw.csv", header=0, delimiter=",")
        return df_raw
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


def clean_data(df, plot):
    if plot:
        # Inspect missing data
        for col in df.columns:
            print('{} - {}%'.format(col, round(np.mean(df[col].isnull()) * 100)))

        # Visualize it with a plot
        ax = sns.heatmap(df[df.columns].isnull())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, rotation_mode='anchor', ha='right')
        #plt.tight_layout()
        #plt.show()

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

    # We don't need: InvoiceNo, InvoiceDate, Description, UnitPrice, Country
    # these might be useful in more complex models
    drop = ["InvoiceNo", "Description"]
    df.drop(drop, inplace=True, axis=1)

    return df

def add_price(df):

    df['Price'] = df['Quantity'] * df['UnitPrice']
    
    # check using CustomerID = 
    # print('')
    # print(df.loc[df['CustomerID'] == 17850.0])
    # filter = df.loc[df['CustomerID'] == 17850.0]
    # total_price = sum(filter['Price'])
    # print(total_price)
   
    return df

def collapse_df(df):
    df2 = df.groupby(['CustomerID']).agg({'Price': ['sum']}).reset_index()

    # check
    # print(df2.loc[df2['CustomerID'] == 17850.0])
   
    return df2

def money_distribution(df, plot):
    lst = []
    for index in range(df.shape[0]):
        lst.append(df.loc[index, 'Price'].values[0])

    #print(df['Price'].squeeze().value_counts())
    #counter=collections.Counter(lst)
    #print(counter)
    if plot == True:
        plt.plot(df['Price'].squeeze())
        plt.show()
    
    # everybody below 25% percentile (305.56 dollars) = low spenders
    # everybody between 25% and 50% percentile (305.56 - 1631.6225) = medium spender
    # everybody above 75% percentile (1631.6225) = high spender
    df['Spender'] = ''
    print(df)
    df['Price'].values[0] = df['Price'].squeeze()
    print(df['Price'])
    #a = df.loc(df['Price'] < 305.56)
    #print(a)

    if df['Price'].values[0] > 1:
        print('shitt')

    #df_collapsed.loc[index, 'Price'].values[0]
    #df['new column name'] = np.where(df['Gender']=='M', 1, 0)

if __name__ == '__main__':
    main()