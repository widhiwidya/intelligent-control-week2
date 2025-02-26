import pandas as pd

# Membaca dataset
dataset = pd.read_csv('colors.csv')

# Menampilkan kolom yang ada dalam dataset
print(dataset.columns)
