import pandas as pd


file_path = r'/data/psychoactiveCompounds_Dataset.csv'
file_path2 = r'C:\Users\Lenovo\Desktop\Psychoactive-Compounds-Analysis\src\model\compoundNames_Scraped.csv'
data1 = pd.read_csv(file_path)
data2 = pd.read_csv(file_path2)

print("Dataset 1 Columns:", data1.columns.tolist())
print()
print("Dataset 2 Columns:", data2.columns.tolist())
print()

overlapping_columns = set(data1.columns).intersection(set(data2.columns))
print("Overlapping Columns:", list(overlapping_columns))
print()

unique_columns_data1 = set(data1.columns) - overlapping_columns
unique_columns_data2 = set(data2.columns) - overlapping_columns
print("Columns unique to Dataset 1:", list(unique_columns_data1))
print()
print("Columns unique to Dataset 2:", list(unique_columns_data2))
print()

rows_data1, cols_data1 = data1.shape
rows_data2, cols_data2 = data2.shape
print("Dataset 1 has", rows_data1, "rows and", cols_data1, "columns")
print("Dataset 2 has", rows_data2, "rows and", cols_data2, "columns")