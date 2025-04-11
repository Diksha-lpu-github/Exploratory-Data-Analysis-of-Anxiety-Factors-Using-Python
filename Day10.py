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

# Financial stress and anxiety
sns.violinplot(x=df["Recent Major Life Event"], y=df["Anxiety Level (1-10)"], hue=df["Recent Major Life Event"], palette="viridis", legend=False)
plt.title("Impact of Financial Stress on Anxiety")
plt.show()

# Meditation effect
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Therapy Sessions (per month)"], y=df["Anxiety Level (1-10)"], palette="magma", hue=df["Therapy Sessions (per month)"], legend=False)
plt.title("Effect of Meditation on Anxiety Levels")
plt.show()

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

# Create a Stress Score
df["stress_score"] = df["Stress Level (1-10)"] + df["Therapy Sessions (per month)"]
print(df[["stress_score", "Anxiety Level (1-10)"]].head())
sns.lmplot(x="stress_score", y="Anxiety Level (1-10)", data=df, hue="Gender", aspect=1.5)
plt.title("Relationship Between Stress Score and Anxiety Level")
plt.show()

# Grouping high-risk individuals
high_risk = df[df["stress_score"] > df["stress_score"].quantile(0.75)]
print("High risk individuals: ",high_risk.head())
