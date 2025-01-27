# lookalike_model.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Data Cleaning and Date Parsing
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Step 2: Feature Engineering
# Calculate the days since customer signup
customers['SignupDays'] = (pd.to_datetime('today') - customers['SignupDate']).dt.days

# Grouping transaction data for each customer
customer_transactions = transactions.groupby('CustomerID').agg(
    total_spent=('TotalValue', 'sum'),
    avg_transaction_value=('TotalValue', 'mean'),
    total_transactions=('TransactionID', 'count')
).reset_index()

# Merge the transaction data with customer profile data
customers = pd.merge(customers, customer_transactions, on='CustomerID', how='left')

# Step 3: Check for missing values
print(customers.isnull().sum())  # Check how many missing values are in the columns

# Step 4: Handle missing values - Fill missing values with 0
customers['total_spent'].fillna(0, inplace=True)
customers['avg_transaction_value'].fillna(0, inplace=True)
customers['total_transactions'].fillna(0, inplace=True)

# Step 5: Prepare features for similarity calculation
features = ['SignupDays', 'total_spent', 'avg_transaction_value', 'total_transactions']

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(customers[features])

# Step 6: Calculate Cosine Similarity
similarity_matrix = cosine_similarity(normalized_features)

# Step 7: Recommend Top 3 Similar Customers for C0001 to C0020
lookalikes = {}

for i in range(20):  # For customers C0001 to C0020
    customer_id = customers['CustomerID'][i]
    similarity_scores = similarity_matrix[i]  # Similarity scores for this customer
    
    # Get indices of the top 3 most similar customers (excluding the customer itself)
    similar_indices = similarity_scores.argsort()[-4:-1][::-1]  # Exclude the customer itself (index 0)
    
    # Get the corresponding customer IDs and similarity scores
    similar_customers = [(customers['CustomerID'][idx], similarity_scores[idx]) for idx in similar_indices]
    
    # Add to lookalikes dictionary
    lookalikes[customer_id] = similar_customers

# Step 8: Convert the results into a DataFrame and save as Lookalike.csv
lookalike_data = []
for cust_id, similar_customers in lookalikes.items():
    for similar_cust_id, score in similar_customers:
        lookalike_data.append([cust_id, similar_cust_id, score])

lookalike_df = pd.DataFrame(lookalike_data, columns=['CustomerID', 'LookalikeID', 'SimilarityScore'])

# Save the Lookalike Model to a CSV file
lookalike_df.to_csv('Lookalike.csv', index=False)

print("Lookalike model completed and saved to Lookalike.csv")
