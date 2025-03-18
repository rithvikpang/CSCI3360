import pandas as pd
import os

# Define input and output file paths
input_file = "data/amazon.csv"
output_dir = "filtered_data"
output_file = os.path.join(output_dir, "amazon_filtered.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Data cleaning functions
def clean_price(price):
    """Convert price string to numeric value."""
    return float(price.replace("₹", "").replace(",", "").strip()) if isinstance(price, str) and "₹" in price else None

def clean_percentage(percentage):
    """Convert percentage string to numeric value."""
    return float(percentage.replace("%", "").strip()) if isinstance(percentage, str) and "%" in percentage else None

def clean_rating(rating):
    """Convert rating string to float."""
    return float(rating) if rating.replace(".", "", 1).isdigit() else None

def clean_count(count):
    """Convert count string to integer."""
    return int(count.replace(",", "").strip()) if isinstance(count, str) and count.replace(",", "").isdigit() else None

# Load the dataset
df = pd.read_csv(input_file)

# Apply cleaning functions
df["discounted_price"] = df["discounted_price"].apply(clean_price)
df["actual_price"] = df["actual_price"].apply(clean_price)
df["discount_percentage"] = df["discount_percentage"].apply(clean_percentage)
df["rating"] = df["rating"].apply(clean_rating)
df["rating_count"] = df["rating_count"].apply(clean_count)

# Remove rows with missing critical values
df = df.dropna(subset=["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"])

# Convert prices from INR to USD using today's exchange rate
# As of March 17, 2025, 1 INR = 0.011505 USD
inr_to_usd_rate = 0.011505
df["discounted_price_usd"] = df["discounted_price"] * inr_to_usd_rate
df["actual_price_usd"] = df["actual_price"] * inr_to_usd_rate

# Filter trending products (Example: rating ≥ 4.0 and discount ≥ 30%)
filtered_df = df[(df["rating"] >= 4.0) & (df["discount_percentage"] >= 30)]

# Save filtered data
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")