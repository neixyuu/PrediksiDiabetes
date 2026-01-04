import pandas as pd

# Load dataset
df = pd.read_csv('diabetes.csv')

# Proses penghitungan
median = df['SkinThickness'][df['SkinThickness'] != 0].median()
print(f"Median: {median}")

# Output
print(f"Mean: {df['SkinThickness'][df['SkinThickness'] != 0].mean():.1f}")
print(f"Min: {df['SkinThickness'][df['SkinThickness'] != 0].min()}")
print(f"Max: {df['SkinThickness'][df['SkinThickness'] != 0].max()}") 