from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv

options = webdriver.ChromeOptions()
options.add_argument('--headless')  # For background processing
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def extract_cids(base_url, total_pages):
    cids = []
    driver.get(base_url)

    for page in range(total_pages):
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/compound/']")))
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/compound/']")
        unique_urls = {link.get_attribute('href') for link in links if "#section=" not in link.get_attribute('href')}

        for url in unique_urls:
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'PubChem CID')]")))
            cid_element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'PubChem CID')]/following-sibling::div")))
            cid = cid_element.text.strip()
            cids.append(cid)
            print(f"Visited URL: {url} Extracted CID: {cid}")

        # More robust handling for clicking the 'Next' button
        try:
            next_button = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.page_link.next"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
            next_button.click()
            print(f"Clicked on the next button to go to page {page + 2}")
        except Exception as e:
            print(f"Failed to find or click next button: {str(e)}")
            break

    driver.quit()
    return cids


base_url = "https://www.ncbi.nlm.nih.gov/pccompound?cmd=HistorySearch&hinit=true&query_key=10&WebEnv=MCID_66a1284318636418fb483847"
total_pages = 2  # Adjust as necessary

all_cids = extract_cids(base_url, total_pages)

# Save the CIDs to a CSV file
csv_filename = "pubchem_cids.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Compound CID"])
    for cid in all_cids:
        writer.writerow([cid])

print(f"Total CIDs collected: {len(all_cids)}")
print(f"CIDs have been written to {csv_filename}")
