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
    df_clean = clean_data(df_raw)
    df_with_price = add_price(df_clean)
    df_collapsed = collapse_df(df_with_price)
    print(df_collapsed.describe())
    

    print('')
    for index in range(4339):
        if df_collapsed.loc[index, 'Price'].values[0] < 1:
            print('errorr')
            print(df_collapsed.loc[index, 'CustomerID'])

    print('')
    money_distribution(df_collapsed)

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


def clean_data(df):
    # Inspect missing data
    for col in df.columns:
        print('{} - {}%'.format(col, round(np.mean(df[col].isnull()) * 100)))

   
   
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
    #drop = ["InvoiceNo", "InvoiceDate", "Description", "UnitPrice", "Country"]
    #df.drop(drop, inplace=True, axis=1)

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

def money_distribution(df):
    lst = []
    for index in range(4339):
        lst.append(df.loc[index, 'Price'].values[0])
    #print(lst)

    print(df['Price'].squeeze().value_counts())
    
    #counter=collections.Counter(lst)
    #print(counter)
    plt.plot(df['Price'].squeeze())
    plt.show()

if __name__ == '__main__':
    main()