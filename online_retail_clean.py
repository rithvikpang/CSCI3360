"""Module to handle the cleaning of data."""

import os
import pandas as pd

df = pd.read_csv('data/online_retail_dataset.csv')
output_dir = "filtered_data"
output_file = os.path.join(output_dir, "online_retail_dataset.csv")


# Modifies the data inplace to drop rows with any None values

df = df.dropna()

# NOTE: There are no duplicates in this dataset but leaving the code here.
df = df.drop_duplicates()

# Ensuring proper date format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed')

# Ensuring proper data type format.
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df['StockCode'] = df['StockCode'].astype(str)

print(df["Quantity"].describe())  # Check min/max for outliers
df = df[df["Quantity"] >= 0]

# NOTE: Can change this later.
df = df[df["Quantity"] < 100]
print(len(df))

filtered_df = df
filtered_df.to_csv(output_file, index=False)
print(f"Filtered data saved to {output_file}")
















