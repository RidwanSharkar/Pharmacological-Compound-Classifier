# backend/s3_services/s3_client.py:

import boto3


def upload_file_to_s3(file_name, bucket, object_name=None):
    s3_client = boto3.client('s3')
    if object_name is None:
        object_name = file_name
    response = s3_client.upload_file(file_name, bucket, object_name)
    return response


if __name__ == "__main__":
    upload_file_to_s3('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\backend\\temp\\compoundClassifications_scraped.parquet', 'molecular-and-pharmacological-data', 'parquet/compoundClassifications_scraped.parquet')
    upload_file_to_s3('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\backend\\temp\\computedParameters_scraped.parquet', 'molecular-and-pharmacological-data', 'parquet/computedParameters_scraped.parquet')
    upload_file_to_s3('C:\\Users\\Lenovo\\Desktop\\Pharmacological-Chemical-Compound-Classifier\\backend\\temp\\compoundCIDs_scraped.parquet', 'molecular-and-pharmacological-data', 'parquet/compoundCIDs_scraped.parquet')