# backend/s3_services/parquet_handler.py
import pandas as pd


def csv_to_parquet(source_path, dest_path):
    df = pd.read_csv(source_path)
    df.to_parquet(dest_path)


if __name__ == "__main__":
    csv_to_parquet('../data/compoundClassifications_scraped.csv', '../temp/compoundClassifications_scraped.parquet')
    csv_to_parquet('../data/computedParameters_scraped.csv', '../temp/computedParameters_scraped.parquet')
