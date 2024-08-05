# backend/s3_services/parquet_handler.py:

import pandas as pd


def csv_to_parquet(source_path, dest_path):
    df = pd.read_csv(source_path)
    df.to_parquet(dest_path)


if __name__ == "__main__":
    base_path = 'C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\backend\\temp\\'
    csv_to_parquet('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\compoundClassifications_scraped.csv', base_path + 'compoundClassifications_scraped.parquet')
    csv_to_parquet('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\computedParameters_scraped.csv', base_path + 'computedParameters_scraped.parquet')
    csv_to_parquet('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\data\\compoundCIDs_scraped.csv', base_path + 'compoundCIDs_scraped.parquet')