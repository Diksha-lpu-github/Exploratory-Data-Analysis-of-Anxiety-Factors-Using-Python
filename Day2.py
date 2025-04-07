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
