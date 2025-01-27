"""
eCommerce Data Science Assignment

Overview:
This script contains the solution for the eCommerce Data Science Assignment, which focuses on three key tasks:
1. Exploratory Data Analysis (EDA)
2. Lookalike Model
3. Customer Segmentation using Clustering

Tasks Breakdown:
1. Exploratory Data Analysis (EDA):
   - Perform data cleaning and analysis on the provided datasets (Customers.csv, Products.csv, Transactions.csv).
   - Derive actionable business insights such as spending patterns, regional distribution, and product performance.

2. Lookalike Model:
   - Develop a model that recommends the top 3 similar customers based on their profile and transaction history.
   - Use customer and product features and calculate similarity scores between customers.

3. Customer Segmentation / Clustering:
   - Apply clustering techniques (e.g., K-Means) to segment customers based on their transaction and profile data.
   - Evaluate the clusters using metrics like DB Index and visualize the results.

File Structure:
    - 'FirstName_LastName_EDA.py' : Python script for Exploratory Data Analysis.
    - 'FirstName_LastName_Lookalike.py' : Python script for Lookalike Model.
    - 'FirstName_LastName_Clustering.py' : Python script for Customer Segmentation.
    - 'FirstName_LastName_Lookalike.csv' : CSV file with top 3 lookalikes and similarity scores for customers.
    - 'FirstName_LastName_EDA.pdf' : Report containing EDA results and insights.
    - 'FirstName_LastName_Clustering.pdf' : Report with clustering results, including metrics and visuals.

How to Run:
1. Clone this repository:
   git clone https://github.com/Khushi-2005/Ecommerce_Analysis.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Python scripts:
   - EDA: python FirstName_LastName_EDA.py
   - Lookalike Model: python FirstName_LastName_Lookalike.py
   - Clustering: python FirstName_LastName_Clustering.py

Insights and Results:
    - **EDA Insights**: Key business insights from analyzing customer, product, and transaction data.
    - **Lookalike Model**: Recommends the top 3 similar customers for each customer based on transaction history and customer profile.
    - **Clustering**: Segments customers into different groups based on their spending and profile data, evaluated using the DB Index.

License:
MIT License
"""
