import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"D:\prodigy_ds_02\titanic.csv")

# Data Cleaning

# Fill missing values in 'Age' with median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill the single missing value in 'Fare' with median fare
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Create a new column 'CabinKnown' indicating whether a cabin number was provided
data['CabinKnown'] = data['Cabin'].notna().astype(int)

# Drop the original 'Cabin' column
data.drop(columns=['Cabin'], inplace=True)

# Verify that there are no more missing values
print("Missing values after cleaning:\n", data.isnull().sum())

# Descriptive Statistics

# Generate summary statistics for numerical columns
summary_statistics = data.describe()
print("\nSummary Statistics:\n", summary_statistics)

# Visualizing Distributions

# Histogram for 'Age'
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram for 'Fare'
plt.figure(figsize=(10, 6))
sns.histplot(data['Fare'], kde=True, bins=30)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Bar chart for 'Survived'
plt.figure(figsize=(8, 5))
sns.countplot(data['Survived'])
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Analyzing Relationships

# Scatter plot for 'Age' vs 'Fare'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', data=data)
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Box plot for 'Pclass' vs 'Fare'
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=data)
plt.title('Passenger Class vs Fare')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

#Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Identifying Patterns and Trends

# Survival rates based on different passenger classes
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Survival rates based on gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

# Age distribution by survival
plt.figure(figsize=(10, 6))
sns.histplot(data[data['Survived'] == 1]['Age'], kde=True, color='green', label='Survived')
sns.histplot(data[data['Survived'] == 0]['Age'], kde=True, color='red', label='Not Survived')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()
