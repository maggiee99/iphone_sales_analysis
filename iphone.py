import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('iphone_sales.csv')

# Display the first few rows of the dataframe
print(df.head())

# Basic Data Exploration
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handling missing values (example: fill with median)
df = df.fillna(df.median())

# Visualize sales by model
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Sales', data=df)
plt.title('iPhone Sales by Model')
plt.xlabel('Model')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()

# Feature Selection and Target Variable
# Assuming features like 'Price', 'MarketingSpend', 'EconomicIndicator', etc.
X = df[['Price', 'MarketingSpend', 'EconomicIndicator']]  # replace with actual feature names
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Plotting actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
