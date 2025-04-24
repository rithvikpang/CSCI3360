import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('filtered_data/combined_cleaned_data.csv', low_memory=False)

# Fill necessary NaNs before calculating demand_score
df['Quantity'] = df['Quantity'].fillna(0)
df['Sell Price'] = df['Sell Price'].fillna(df['Sell Price'].mean())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

# Create demand score and levels
df['demand_score'] = (
    df['Quantity'] + 
    df['Sell Price'] / df['Sell Price'].mean() * 1.5 + 
    df['Rating'] * 2
)

df['demand_level'] = pd.qcut(
    df['demand_score'], q=[0, .34, .67, 1], labels=['low', 'medium', 'high']
)

# Date features
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Feature categories
cat_cols = ['Category', 'Season', 'Location', 'Month', 'DayOfWeek', 'Year']
num_cols = ['Quantity', 'Sell Price', 'Discount Price', 'Discount %', 'Rating']
dropped_cols = ['Purchase ID', 'Item Title', 'Description', 'Date', 'Review', 'demand_score', 'demand_level', 'Dataset']

# Use a 10k sample for PCA visualization
df_sample = df.sample(n=10000, random_state=42)
X = df_sample.drop(dropped_cols, axis=1)
y = df_sample['demand_level']

# Preprocessing pipelines
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Transform features and apply PCA
X_processed = preprocessor.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualization: Decision boundaries
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

clf_numeric = LogisticRegression(max_iter=1000)
clf_numeric.fit(X_train, y_train.map({'low': 0, 'medium': 1, 'high': 2}))
Z = clf_numeric.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.Categorical(y).codes, edgecolors='k', cmap='coolwarm', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Logistic Regression Decision Regions (PCA-reduced, 10k Sample)')
plt.legend(handles=scatter.legend_elements()[0], labels=pd.Categorical(y).categories.tolist())
plt.grid(True)
plt.show()
