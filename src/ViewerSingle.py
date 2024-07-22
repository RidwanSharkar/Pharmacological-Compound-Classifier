import pandas as pd


file_path = r'C:\Users\Lenovo\Desktop\Psychoactive-Compounds-Analysis\data\test_rows_labels.csv'
data = pd.read_csv(file_path)

print("Dataset 1 Columns:", data.columns.tolist())
print()

rows_data, cols_data = data.shape
print("Dataset 1 has", rows_data, "rows and", cols_data, "columns")
print(data.head(1).to_string())
