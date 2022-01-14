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
    
    # add variables 
    df_with_price = add_columns(df_clean)
    #print(df_with_price[1200:1250])
    



    # collapse data
    df_collapsed = collapse_df(df_with_price)
    print(df_collapsed.describe())
    

    print('')
    for index in range(df_collapsed.shape[0]):
        if df_collapsed.loc[index, 'Price'] < 1:
            print('errorr')
            print(df_collapsed.loc[index, 'CustomerID'])

    print('')
    df_maybe = money_distribution(df_collapsed, False)
    print(df_maybe)


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

def add_columns(df):

    df['Price'] = df['Quantity'] * df['UnitPrice']

    print(df['Country'].value_counts())
    
    codes, uniques = pd.factorize(df['Country'])
    print(len(codes))
    df['CountryCode'] = codes

    print('hier')
    df['WorkingHours'] = ''
    for _, row in df.iterrows():
        if int(row['InvoiceDate'][-8:-6]) <= 17 and int(row['InvoiceDate'][-8:-6]) >= 9:
            df['WorkingHours'] = 1
        else:
            df['WorkingHours'] = 0

    print(df[100:130])
   
    return df


def collapse_df(df):
    df2 = df.groupby(['CustomerID'])['Price'].sum().reset_index()
    df3 = df.groupby(['CustomerID'])['CountryCode'].max().reset_index()
    
    df4 = df.groupby(['CustomerID'])['WorkingHours'].max().reset_index()
    # check
    #print(df3['CountryCode'].value_counts())
    #print(df2.loc[df2['CustomerID'] == 17850.0])

    df_final_temp = df2.merge(df3, left_on='CustomerID', right_on='CustomerID')
    df_final = df_final_temp.merge(df4, left_on='CustomerID', right_on='CustomerID')

    return df_final

def money_distribution(df, plot):
    lst = []
    for index in range(df.shape[0]):
        lst.append(df.loc[index, 'Price'])

    #print(df['Price'].squeeze().value_counts())
    #counter=collections.Counter(lst)
    #print(counter)
    if plot == True:
        plt.plot(df['Price'].squeeze())
        plt.show()
    

    df['Spender'] = ''
    
    df['Spender'] = df.apply(Spender, axis = 1)
    
    return df

def Spender(df):   
    # everybody below 25% percentile (305.56 dollars) = low spenders
    # everybody between 25% and 50% percentile (305.56 - 1631.6225) = medium spender
    # everybody above 75% percentile (1631.6225) = high spender
    if df['Price'] <= 305.56:
        return 1
    elif df['Price'] > 305.56 and df['Price'] < 1631.6225:
        return 2
    elif df['Price'] > 1631.6225:
        return 3
    else:
        return np.nan
    


if __name__ == '__main__':
    main()