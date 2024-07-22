import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_pubchem_properties(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Assuming you've identified the correct table or section id/class
    properties_table = soup.find('table', {'id': 'Computed_Properties_Table'})

    # Extract data from the table
    properties = {}
    if properties_table:
        rows = properties_table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 1:
                property_name = cols[0].text.strip()
                property_value = cols[1].text.strip()
                properties[property_name] = property_value

    return properties


# Example usage
cid = 122130742  # Example CID
properties = scrape_pubchem_properties(cid)
properties_df = pd.DataFrame([properties])  # Convert to DataFrame for further processing

# Now, properties_df can be used to predict activities using your trained model
