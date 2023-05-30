#import libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the customer data from CSV file
data = pd.read_csv('CustomerLoyaltyCardData.csv')

# Task 1: Display the first 10 rows of the dataset
print("First 10 rows of the dataset:")
print(data.head(10))
print()

# Task 2: Get the basic statistics of the dataset
print("Basic statistics of the dataset:")
print(data.describe())
print()

# Task 3: Visualize the distribution of customers by gender
sns.countplot(data=data, x='Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Customers by Gender')
plt.show()

# Task 4: Visualize the age distribution of customers
sns.histplot(data=data, x='Age', bins=10)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Customers')
plt.show()

# Task 5: Visualize the annual income distribution of customers
sns.histplot(data=data, x='Annual Income (k$)', bins=10)
plt.xlabel('Annual Income ($k)')
plt.ylabel('Count')
plt.title('Annual Income Distribution of Customers')
plt.show()

# Task 6: Visualize the spending score distribution of customers
sns.histplot(data=data, x='Spending Score (1-100)', bins=10)
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.title('Spending Score Distribution of Customers')
plt.show()
