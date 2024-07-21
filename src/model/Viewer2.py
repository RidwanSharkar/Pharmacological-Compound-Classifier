import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

file_path = r'C:\Users\Lenovo\Desktop\Psych Analysis\data\psychoactive compounds.csv'
df = pd.read_csv(file_path)
# print("Dataset 1 Columns:", data.columns.tolist())

pubchem_cids = df['PubChem Compound CID'].tolist()


def get_compound_name(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            # The title in PubChem pages includes the compound name
            title = soup.title.text
            # Usually, the title format is "Compound Name | PubChem", so split by '|'
            compound_name = title.split('|')[0].strip()
            return compound_name
        else:
            print(f"Failed to retrieve data for CID {cid}: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#################################################
compound_names = {}

for cid in pubchem_cids:
    name = get_compound_name(cid)
    if name:
        compound_names[cid] = name
    else:
        compound_names[cid] = "Not Found"

# Convert the dictionary to a DataFrame
names_df = pd.DataFrame(list(compound_names.items()), columns=['PubChem Compound CID', 'Compound Name'])

# Optionally, save the results to a CSV file
names_df.to_csv('compound_names.csv', index=False)
