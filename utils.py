import boto3
import os
import pandas as pd
from io import BytesIO
from langchain.vectorstores import DeepLake
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings

ACCESS_KEY = 'xxxxxxxxxxx'
SECRET_KEY = 'xxxxxxxxxxxxxxxxxxxxxx'

def download_from_s3(bucket_name, source_blob_name):
    # Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    # Read the file from S3
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=source_blob_name)
        data = obj['Body'].read()
        csv_data = pd.read_csv(BytesIO(data)) 
    
    except Exception as e:
        print("Error reading file from S3:", e)
    
    return csv_data


def upload_to_s3(bucket_name, source_file_name, destination_blob_name):
    # Explicitly use service account credentials by specifying the private key file
    # Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    # Upload the file to S3
    try:
        s3.upload_file(source_file_name, bucket_name, destination_blob_name)
        print(f"File '{destination_blob_name}' uploaded successfully to '{bucket_name}' bucket.")

    except Exception as e:
        print("Error uploading file to S3:", e)

def open_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            job_description = file.read()
            return job_description
            # print(job_description)  # Or do something else with the contents
    except FileNotFoundError:
        print("The file does not exist or the path is incorrect.")
    except Exception as e:
        print("An error occurred:", e)

def write_text_file(file_path, response):
    try:
        with open(file_path, 'w') as file:
            file.write(response)
        print(f"Text written to {file_path} successfully.")
    except Exception as e:
        print("An error occurred:", e)

def upload_data_to_vector_db(csv_name, activeloop_token, activeloop_org_id, activelooop_dataset_name):
    loader = CSVLoader(file_path=csv_name)
    data = loader.load()
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    os.environ['ACTIVELOOP_TOKEN'] = activeloop_token
    activeloop_org_id= activeloop_org_id
    activelooop_dataset_name = activelooop_dataset_name
    dataset_path = f"hub://{activeloop_org_id}/{activelooop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

    # add documents to our Deep Lake dataset
    db.add_documents(data)
