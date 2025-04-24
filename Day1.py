# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#1
# Load dataset
df = pd.read_csv("enhanced_anxiety_dataset.csv")

# Display basic info
print(df.info())
print(df.head())
