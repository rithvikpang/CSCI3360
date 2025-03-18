import pandas as pd
import os

def load_csv_safely(file_path: str) -> pd.DataFrame:
    """Attempt to read the CSV at file_path."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error reading '{file_path}': {str(e)}")
    return df

def check_columns_exist(df: pd.DataFrame, columns: list, file_label: str):
    """
    Check if each column in 'columns' exists in df. If not, raise KeyError.
    Supports multiple columns in a single string separated by 'AND'.
    """
    for col in columns:
        col = col.strip()
        if not col:
            # empty mapping => we ignore
            continue

        # If 'AND' is in the specification, check and combine.
        if " AND " in col:
            subcols = [c.strip() for c in col.split("AND")]
            for sc in subcols:
                if sc not in df.columns:
                    raise KeyError(f"Error: Column '{sc}' not found in '{file_label}' dataset.")
        else:
            # Single column check
            if col not in df.columns:
                raise KeyError(f"Error: Column '{col}' not found in '{file_label}' dataset.")

def combine_columns(df: pd.DataFrame, col_spec: str) -> pd.Series:
    """
    Return a series corresponding to the final merged column data 
    for a single 'target column' from the given df, based on col_spec:
       - "" => just return an empty string for each row
       - "colA" => return df["colA"]
       - "colA AND colB" => return df["colA"] + ", " + df["colB"]
    """
    col_spec = col_spec.strip()
    if not col_spec:
        # Return an empty string for every row
        return pd.Series([""] * len(df), index=df.index)
    
    if " AND " in col_spec:
        # We have multiple columns to merge, separated by "AND"
        subcols = [c.strip() for c in col_spec.split("AND")]
        merged = df[subcols[0]].astype(str) + ", " + df[subcols[1]].astype(str)
        return merged
    else:
        # Single column
        return df[col_spec].astype(str)

def build_mapped_dataframe(df: pd.DataFrame, dataset_label: str, mapping: dict, final_cols: list) -> pd.DataFrame:
    """Builds a df that includes all columns in final_cols."""

    # Create an empty DataFrame with the same index as 'df'
    out_df = pd.DataFrame(index=df.index)
    
    # Hard-code the dataset label for each row
    out_df["Dataset"] = dataset_label

    for col in final_cols:
        if col == "Dataset":
            continue  # We already set this above
        # If col not in mapping, default is empty string:
        source_col_spec = mapping.get(col, "")
        out_df[col] = combine_columns(df, source_col_spec)

    return out_df


def columns_from_mapping(mapping: dict) -> list:
    """
    Gather all column specs from a dictionary (ignoring empty strings).
    Each mapping value might be a single column name or 'colA AND colB'.
    """
    cols = []
    for spec in mapping.values():
        spec = spec.strip()
        if not spec:
            continue
        # Could have multiple columns joined by 'AND'
        if " AND " in spec:
            cols.extend(c.strip() for c in spec.split("AND"))
        else:
            cols.append(spec)
    return cols

def main():
    # Filepaths
    file1 = os.path.join("filtered_data", "online_retail_dataset.csv")
    file2 = os.path.join("filtered_data", "amazon_filtered.csv")
    file3 = os.path.join("filtered_data", "Filtered_Shopping_Trends.csv")

    # Combined target columns for actual learning 
    final_columns = [
        "Dataset", # Corresponds to which of the 3 datasets the row comes from
        "Purchase ID",
        "Item Title",
        "Category",
        "Description",
        "Quantity",
        "Sell Price",
        "Discount Price",
        "Discount %",
        "Date",
        "Season",
        "Location",
        "Rating",
        "Review",
    ]

    # Columns of interest from original datasets to be mapped to the above columns

    # online_retail_data
    mapping1 = {
        "Purchase ID": "InvoiceNo",
        "Item Title": "Description",
        "Category": "",
        "Description": "",
        "Quantity": "Quantity",
        "Sell Price": "UnitPrice",
        "Discount Price": "",
        "Discount %": "",
        "Date": "InvoiceDate",
        "Season": "",
        "Location": "Country",
        "Rating": "",
        "Review": ""
    }

    # amazon_filtered
    mapping2 = {
        "Purchase ID": "product_id",
        "Item Title": "product_name",
        "Category": "category",
        "Description": "about_product",
        "Quantity": "",
        "Sell Price": "actual_price_usd",
        "Discount Price": "discounted_price_usd",
        "Discount %": "discount_percentage",
        "Date": "",
        "Season": "",
        "Location": "",
        "Rating": "rating",
        "Review": "review_title AND review_content" # Combine the review_title and actual review_content
    }

    # Filtered_Shopping_Trends
    mapping3 = {
        "Purchase ID": "Customer ID",
        "Item Title": "Item Purchased",
        "Category": "Category",
        "Description": "Size AND Color", # Combine Size and Color for the description
        "Quantity": "Number of Items Purchased",
        "Sell Price": "Purchase Amount (USD)",
        "Discount Price": "",
        "Discount %": "",
        "Date": "",
        "Season": "Season",
        "Location": "Location",
        "Rating": "Review Rating",
        "Review": ""
    }

    # Load and check .csvs
    df1 = load_csv_safely(file1)
    df2 = load_csv_safely(file2)
    df3 = load_csv_safely(file3)

    check_columns_exist(df1, columns_from_mapping(mapping1), "online_retail_dataset.csv")
    check_columns_exist(df2, columns_from_mapping(mapping2), "amazon_filtered.csv")
    check_columns_exist(df3, columns_from_mapping(mapping3), "Filtered_Shopping_Trends.csv")

    # Map the columns of interest into the target columns
    mapped_df1 = build_mapped_dataframe(df1, "Retail", mapping1, final_columns)
    mapped_df2 = build_mapped_dataframe(df2, "Amazon", mapping2, final_columns)
    mapped_df3 = build_mapped_dataframe(df3, "Trends", mapping3, final_columns)

    # Combine and organize them
    final_df = pd.concat([mapped_df1, mapped_df2, mapped_df3], ignore_index=True)

    final_df = final_df[final_columns]

    # Export to .csv
    output_file = "filtered_data/combined_cleaned_data.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Successfully created '{output_file}' with {len(final_df)} rows.")

if __name__ == "__main__":
    main()
