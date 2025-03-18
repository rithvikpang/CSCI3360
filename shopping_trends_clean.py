import pandas as pd
import os

# Load the data
file_path = 'data/Shopping_Trends.csv'
df = pd.read_csv(file_path)

# Display basic info and first few rows to understand the dataset
print(df.info())
print(df.head())

# Drop duplicate entries
df = df.drop_duplicates()

# Handle missing values - Drop rows with missing essential fields
df = df.dropna(subset=['Item Purchased', 'Category', 'Purchase Amount (USD)', 'Review Rating', 'Seasonal Trends'])


# Filter products based on a revenue threshold (e.g., top trending products)
threshold = df['Review Rating'].quantile(0.75) # Top 25% highest reviews
filtered_df = df[df['Review Rating'] >= threshold]


# Define output directory and file
output_dir = 'filtered_data'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'Filtered_Shopping_Trends.csv')

# Save the cleaned and filtered data
filtered_df.to_csv(output_file, index=False)

print(f'Filtered data saved to {output_file}')