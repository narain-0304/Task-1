import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display plots inline
#matplotlib inline

# Load dataset from CSV or string (for example purpose, use from clipboard or file)
data = pd.read_csv('Task 1/liver_patient_data.csv') 
 # or use StringIO if it's a string
data['alkphos'].fillna(data['alkphos'].median(), inplace=True)

print(data.info())
print("\nMissing Values:\n", data.isnull().sum())
print("\nDescriptive Stats:\n", data.describe())
print("\nUnique values:\n", data['gender'].value_counts())
# Fill numeric columns with median
data.fillna(data.median(numeric_only=True), inplace=True)

print("\nMissing Values:\n", data.isnull().sum())
# Or use a more selective approach
# data['ag_ratio'].fillna(data['ag_ratio'].median(), inplace=True)

# Convert 'gender' to numeric
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
#print(data.select_dtypes(include='object').columns)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_cols = ['tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']

data[scaled_cols] = scaler.fit_transform(data[scaled_cols])

# Visualize with boxplots
plt.figure(figsize=(15, 8))
sns.boxplot(data=data[scaled_cols])
plt.xticks(rotation=45)
plt.title("Boxplot of Scaled Features")
plt.show()

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

data_cleaned = remove_outliers(data, scaled_cols)

print(data_cleaned.head())
print("Final shape after outlier removal:", data_cleaned.shape)

data_cleaned.to_csv("Liver-Patient-Cleaned.csv", index=False)