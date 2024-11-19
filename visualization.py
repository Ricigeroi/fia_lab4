import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with proper NA values and memory settings
dataset = pd.read_csv('datasets/data.csv', na_values=['Not Provided'], low_memory=False)

# Ensure numerical columns are correctly typed
dataset['BasePay'] = pd.to_numeric(dataset['BasePay'], errors='coerce')
dataset['OvertimePay'] = pd.to_numeric(dataset['OvertimePay'], errors='coerce')
dataset['OtherPay'] = pd.to_numeric(dataset['OtherPay'], errors='coerce')
dataset['Benefits'] = pd.to_numeric(dataset['Benefits'], errors='coerce')
dataset['TotalPay'] = pd.to_numeric(dataset['TotalPay'], errors='coerce')
dataset['TotalPayBenefits'] = pd.to_numeric(dataset['TotalPayBenefits'], errors='coerce')

# 1. Histogram of TotalPay
plt.figure(figsize=(10,6))
plt.hist(dataset['TotalPay'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of TotalPay')
plt.xlabel('TotalPay')
plt.ylabel('Frequency')
plt.show()

# 2. Scatter plot of BasePay vs TotalPay
plt.figure(figsize=(10,6))
plt.scatter(dataset['BasePay'], dataset['TotalPay'], alpha=0.5)
plt.title('BasePay vs TotalPay')
plt.xlabel('BasePay')
plt.ylabel('TotalPay')
plt.show()

# 3. Boxplots of Pay Components
pay_components = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']
dataset_melted = dataset.melt(value_vars=pay_components)
plt.figure(figsize=(10,6))
sns.boxplot(x='variable', y='value', data=dataset_melted)
plt.title('Boxplots of Pay Components')
plt.xlabel('Pay Component')
plt.ylabel('Amount')
plt.show()

# 4. Top 10 Job Titles by Frequency

# Create a DataFrame for top job titles
top_jobs = dataset['JobTitle'].value_counts().head(10)
top_jobs_df = top_jobs.reset_index()
top_jobs_df.columns = ['JobTitle', 'Count']

plt.figure(figsize=(10,6))

# Option 1: Use color parameter
sns.barplot(data=top_jobs_df, x='Count', y='JobTitle', color='skyblue')

# Option 2: Use hue parameter
# sns.barplot(data=top_jobs_df, x='Count', y='JobTitle', hue='JobTitle', palette='viridis', dodge=False, legend=False)

plt.title('Top 10 Job Titles')
plt.xlabel('Number of Employees')
plt.ylabel('Job Title')
plt.show()

# 5. TotalPay over Years
yearly_pay = dataset.groupby('Year')['TotalPay'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(data=yearly_pay, x='Year', y='TotalPay', marker='o')
plt.title('Average TotalPay Over Years')
plt.xlabel('Year')
plt.ylabel('Average TotalPay')
plt.show()

# 6. Distribution of Employment Status

# Create a DataFrame for status counts
status_counts = dataset['Status'].fillna('Not Provided').value_counts()
status_counts_df = status_counts.reset_index()
status_counts_df.columns = ['Status', 'Count']

plt.figure(figsize=(8,6))

# Option 1: Use color parameter
sns.barplot(data=status_counts_df, x='Status', y='Count', color='lightgreen')

# Option 2: Use hue parameter
# sns.barplot(data=status_counts_df, x='Status', y='Count', hue='Status', palette='pastel', dodge=False, legend=False)

plt.title('Distribution of Employment Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Heatmap of Correlation Matrix
numeric_cols = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay', 'TotalPayBenefits']
corr_matrix = dataset[numeric_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()
