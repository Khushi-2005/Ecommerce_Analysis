import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score

# Load data
print("Loading data...")
customers = pd.read_csv('/Users/khushiupadhyay/ecommerce_analysis/Customers.csv')  # Update with actual path
transactions = pd.read_csv('/Users/khushiupadhyay/ecommerce_analysis/Transactions.csv')  # Update with actual path

# Preprocessing the data
# Merge data based on CustomerID
customer_data = pd.merge(customers, transactions, on='CustomerID', how='left')

# Feature Engineering: Calculate relevant features
customer_data['SignupDate'] = pd.to_datetime(customer_data['SignupDate'])
customer_data['SignupDays'] = (pd.to_datetime('today') - customer_data['SignupDate']).dt.days

# Calculate total spent, average transaction value, and total transactions
total_spent = customer_data.groupby('CustomerID')['TotalValue'].sum()
avg_transaction_value = customer_data.groupby('CustomerID')['TotalValue'].mean()
total_transactions = customer_data.groupby('CustomerID').size()

# Join these features back to customer_data
customer_data['total_spent'] = customer_data['CustomerID'].map(total_spent)
customer_data['avg_transaction_value'] = customer_data['CustomerID'].map(avg_transaction_value)
customer_data['total_transactions'] = customer_data['CustomerID'].map(total_transactions)

# Handling missing values
customer_data['total_spent'].fillna(0, inplace=True)
customer_data['avg_transaction_value'].fillna(0, inplace=True)
customer_data['total_transactions'].fillna(0, inplace=True)

# Select relevant features for clustering
X = customer_data[['SignupDays', 'total_spent', 'avg_transaction_value', 'total_transactions']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # You can change the number of clusters between 2-10
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Clustering Evaluation (Davies-Bouldin Index)
db_index = davies_bouldin_score(X_scaled, customer_data['Cluster'])
print(f"DB Index: {db_index}")

# Visualize Clusters
plt.figure(figsize=(10, 6))
plt.scatter(customer_data['total_spent'], customer_data['avg_transaction_value'], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation using KMeans Clustering')
plt.xlabel('Total Spent')
plt.ylabel('Average Transaction Value')
plt.colorbar(label='Cluster')
plt.show()

# Save the clustered data to a new CSV
customer_data.to_csv('/Users/khushiupadhyay/ecommerce_analysis/Clustered_Customers.csv', index=False)  # Update with actual path

# Output the number of clusters formed
print(f"Number of clusters formed: {len(customer_data['Cluster'].unique())}")
