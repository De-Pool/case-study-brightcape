import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import csv


def main():
    # Process data
    filename = './data/data-raw.xlsx'
    df_raw = process_data(filename=filename, savedcsv=True)

    # Clean data
    cleaned_dataframe = clean_data(df_raw)


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

    return df


if __name__ == '__main__':
    main()
