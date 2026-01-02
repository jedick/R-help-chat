import requests
import zipfile
import shutil
import boto3
import os


def download_file_from_bucket(bucket_name, s3_key, output_file):
    """Download file from S3 bucket"""

    # https://thecodinginterface.com/blog/aws-s3-python-boto3
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_ACCESS_KEY_SECRET"),
        region_name="us-east-1",
    )
    s3_resource = session.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    bucket.download_file(Key=s3_key, Filename=output_file)


def download_dropbox_file(shared_url, output_file):
    """Download file from Dropbox"""

    # Modify the shared URL to enable direct download
    direct_url = shared_url.replace(
        "www.dropbox.com", "dl.dropboxusercontent.com"
    ).replace("?dl=0", "")

    # Send a GET request to the direct URL
    response = requests.get(direct_url, stream=True)

    if response.status_code == 200:
        # Write the content to a local file
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully as '{output_file}'")
    else:
        print(f"Failed to download file. HTTP Status Code: {response.status_code}")


def download_data():
    """Download the email database"""

    if not os.path.exists("db.zip"):
        # For S3 (need AWS_ACCESS_KEY_ID and AWS_ACCESS_KEY_SECRET)
        # db_20250801.zip: chromadb==1.0.13
        # db_20250801a.zip: chromadb==0.6.3
        download_file_from_bucket("r-help-chat", "db_20260102.zip", "db.zip")
        ## For Dropbox (shared file - key is in URL)
        # shared_link = "https://www.dropbox.com/scl/fi/jx90g5lorpgkkyyzeurtc/db.zip?rlkey=wvqa3p9hdy4rmod1r8yf2am09&st=l9tsam56&dl=0"
        # output_filename = "db.zip"
        # download_dropbox_file(shared_link, output_filename)


def extract_data():
    """Extract the db.zip file"""

    file_path = "db.zip"
    extract_to_path = "./"
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)
