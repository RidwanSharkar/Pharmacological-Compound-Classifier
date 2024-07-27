# backend/s3_services/s3_client.py
import boto3


def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket"""
    s3_client = boto3.client('s3')
    if object_name is None:
        object_name = file_name
    response = s3_client.upload_file(file_name, bucket, object_name)
    return response


if __name__ == "__main__":
    # Example usage: Upload the Parquet files to S3
    upload_file_to_s3('../temp/compoundClassifications_scraped.parquet', 'your-bucket-name', 'parquet/compoundClassifications_scraped.parquet')
    upload_file_to_s3('../temp/computedParameters_scraped.parquet', 'your-bucket-name', 'parquet/computedParameters_scraped.parquet')