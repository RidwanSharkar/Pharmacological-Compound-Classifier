import pandas as pd
import requests
import time
import json

file_path = r'C:\Users\Lenovo\Desktop\Psychoactive-Compounds-Analysis\data\psychoactive compounds.csv'
df = pd.read_csv(file_path)


def get_compound_info(cid):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
    url = f"{base_url}/data/compound/{cid}/JSON"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract compound name
        compound_name = data['Record']['RecordTitle']

        # Extract MeSH Pharmacological Classifications
        compound_activities = []
        sections = data['Record'].get('Section', [])
        for section in sections:
            if section.get('TOCHeading') == 'Pharmacology and Biochemistry':
                for subsection in section.get('Section', []):
                    if subsection.get('TOCHeading') == 'MeSH Pharmacological Classification':
                        for info in subsection.get('Information', []):
                            if "Name" in info:
                                activity_name = info['Name']
                                if activity_name not in compound_activities:
                                    compound_activities.append(activity_name)

        return compound_name, list(set(compound_activities))  # Remove duplicates
    except requests.RequestException as e:
        print(f"Request failed for CID {cid}: {e}")
        return f"Unknown (CID: {cid})", []
    except Exception as e:
        print(f"An error occurred for CID {cid}: {e}")
        return f"Unknown (CID: {cid})", []


# Process each CID
compound_info = {}
for cid in df['PubChem Compound CID'].tolist():
    name, activities = get_compound_info(cid)
    compound_info[cid] = {'name': name, 'activities': activities}
    print(f"CID {cid}: Name = {name}, Activities = {activities}")  # Debug print
    time.sleep(0.2)  # Short delay to avoid overwhelming the API

# Convert to DataFrame
info_df = pd.DataFrame.from_dict(compound_info, orient='index')
info_df.reset_index(inplace=True)
info_df.columns = ['PubChem Compound CID', 'Compound Name', 'Activities']
info_df.to_csv('compound_info.csv', index=False)
