# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("enhanced_anxiety_dataset.csv")

# Display basic info
print(df.info())
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill or drop missing values based on data type
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)  # For numerical columns
df.fillna(df.mode().iloc[0], inplace=True)  # For categorical columns

# Remove duplicates
df.drop_duplicates(inplace=True)

# Verify changes
print(df.info())

# Summary of numerical columns
print(df.describe())

# Unique values in categorical columns
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Count anxiety levels
plt.figure(figsize=(8,5))
sns.countplot(x=df["Anxiety Level (1-10)"], hue=df["Anxiety Level (1-10)"], palette="coolwarm", legend=False)
plt.title("Distribution of Anxiety Levels")
plt.show()

# Boxplot for anxiety levels by age
plt.figure(figsize=(10,5))
sns.boxplot(x=df["Anxiety Level (1-10)"], y=df["Age"], palette="mako", hue=df["Anxiety Level (1-10)"], legend=False)
plt.title("Anxiety Levels Across Different Age Groups")
plt.show()

# Bar plot for gender-wise anxiety distribution
plt.figure(figsize=(8,5))
sns.countplot(x=df["Gender"], hue=df["Anxiety Level (1-10)"], palette="viridis")
plt.title("Anxiety Levels by Gender")
plt.show()

# Sleep pattern vs anxiety
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Anxiety Level (1-10)"], y=df["Sleep Hours"], hue=df["Anxiety Level (1-10)"], palette="coolwarm", legend=False)
plt.title("Impact of Sleep Hours on Anxiety Levels")
plt.show()

# Work stress impact
plt.figure(figsize=(8,5))
sns.countplot(x=df["Stress Level (1-10)"], hue=df["Anxiety Level (1-10)"], palette="rocket")
plt.title("Anxiety Levels Based on Work Stress")
plt.show()
