import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

file_path1 = r'C:\Users\Lenovo\Desktop\Psych Analysis\data\psychs.csv'
file_path2 = r'C:\Users\Lenovo\Desktop\Psych Analysis\data\psychoactive.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

###################################
