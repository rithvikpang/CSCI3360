"""Module to train our data."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = pd.read_csv('filtered_data/combined_cleaned_data.csv', low_memory=False)
#df.dropna(inplace=True)

# Fill necessary NaNs before calculating demand_score
df['Quantity'] = df['Quantity'].fillna(0)
df['Sell Price'] = df['Sell Price'].fillna(df['Sell Price'].mean())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

# NOTE: We can change the numbers as needed, meant to normalize the data. 
df['demand_score'] = (
    df['Quantity'] + 
    df['Sell Price'] / df['Sell Price'].mean() * 1.5 + 
    df['Rating'] * 2
)

df['demand_level'] = pd.qcut(
    df['demand_score'], q=[0, .34, .67, 1], labels=['low', 'medium', 'high']
)

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek

cat_cols = ['Category', 'Season', 'Location', 'Month', 'DayOfWeek', 'Year']
num_cols = ['Quantity', 'Sell Price', 'Discount Price', 'Discount %', 'Rating']

# NOTE: Columns we dont need to train on 
dropped_cols = ['Purchase ID', 'Item Title', 'Description', 'Date', 'Review', 'demand_score', 'demand_level', 'Dataset']

# NOTE: Define logistical regression steps

X = df.drop(dropped_cols, axis=1)
y = df['demand_level']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])


reg_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, multi_class='multinomial'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
reg_model.fit(X_train, y_train)

# Make predictions
y_pred = reg_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



