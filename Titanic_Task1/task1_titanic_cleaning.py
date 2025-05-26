# -*- coding: utf-8 -*-
"""
Created on Mon May 26 14:43:39 2025

@author: capl2
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r"C:\Users\capl2\OneDrive\Pictures\Documents\Titanic_Task1\Titanic-Dataset.csv")

# Display first few rows
print(df.head())

# Check shape
print("Shape of dataset:", df.shape)

# Summary of dataset
df.info()

# Basic statistics
print(df.describe(include='all'))

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop irrelevant or mostly empty columns
df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Detect and remove outliers using boxplot
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

# Remove extreme outliers (after standardization, use z-score thresholds ideally)
df = df[df['Fare'] < 3]  # You can adjust this based on your plot

# Plot count of survivors
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Show correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Save the cleaned dataset
df.to_csv(r"C:\Users\capl2\OneDrive\Pictures\Documents\Titanic_Task1\cleaned_titanic.csv", index=False)

