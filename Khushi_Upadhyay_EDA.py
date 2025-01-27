import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Display basic info about the datasets
print(customers.info())
print(products.info())
print(transactions.info())

# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# Customer region distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=customers['Region'].value_counts().index, y=customers['Region'].value_counts().values, palette='viridis')
plt.title('Customer Distribution by Region')
plt.ylabel('Number of Customers')
plt.xlabel('Region')
plt.show()

# Product price distribution
plt.figure(figsize=(8, 5))
sns.histplot(products['Price'], kde=True, color='teal')
plt.title('Product Price Distribution')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()

# Total sales per month
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions['YearMonth'] = transactions['TransactionDate'].dt.to_period('M')

monthly_sales = transactions.groupby('YearMonth').sum()['TotalValue']
plt.figure(figsize=(10, 6))
monthly_sales.plot(marker='o', color='blue')
plt.title('Monthly Sales Trend')
plt.ylabel('Total Sales (USD)')
plt.xlabel('Month')
plt.grid()
plt.show()

# Top 10 products by revenue
top_products = transactions.groupby('ProductID').sum()['TotalValue'].sort_values(ascending=False).head(10)
top_product_names = products.set_index('ProductID').loc[top_products.index]['ProductName']

plt.figure(figsize=(10, 6))
sns.barplot(x=top_product_names, y=top_products.values, palette='coolwarm')
plt.title('Top 10 Products by Revenue')
plt.xlabel('Product Name')
plt.ylabel('Total Revenue (USD)')
plt.xticks(rotation=45)
plt.show()

# Saving outputs (optional)
# insights = {
#     'Sales Trends': monthly_sales,
#     'Top Products': top_products
# }
# insights.to_csv('eda_insights.csv')  # Save any results if needed
