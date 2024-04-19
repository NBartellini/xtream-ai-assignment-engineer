# from google.cloud import storage
# from google.cloud import bigquery
import pandas as pd
# from google.cloud import exceptions as gcloud_exceptions

# PROJECT_ID = os.environ.get("PROJECT_ID")
# storage_client = storage.Client(project=PROJECT_ID)
# db = bigquery.Client(project=PROJECT_ID)

# def download_file_from_blob(bucket_name: str, source_blob_name: str, destination_file_name: str)->pd.DataFrame:
    # """
    #     Downloads a blob from the specified bucket.

    #     Parameters:
    #         - bucket_name (str): The name of the bucket.
    #         - source_blob_name (str): The name of the source blob.
    #         - destination_file_name (str): The name of the destination file to save the blob.

    #     Returns:
    #         - bool: True if the download is successful, False otherwise.

    #     Raises:
    #        - gcloud_exceptions.NotFound: If the specified bucket or blob is not found.
    # """
#     try:
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(source_blob_name)
#         blob.download_to_filename(destination_file_name)
#         return True
#     except Exception as error:
#         raise gcloud_exceptions.NotFound()

# def ingest_bigquery_data()->pd.DataFrame:
    # """
    #     Ingests data from BigQuery and returns it as a DataFrame.

    #     Returns:
    #         - pd.DataFrame: The ingested data as a DataFrame.
    # """
#     try:
#         query_string =f"""
#             SELECT * FROM table_in_bigquery
#             ORDER BY
#             id
#             """
#         data = db.query(query_string).result().to_dataframe(create_bqstorage_client=False)
#         return data
#     except Exception as error:
#         raise error

def ingest_data(type_download: str, data_path: str)->pd.DataFrame:
    """
        Ingests data from different sources based on the specified type of download.

        Args:
            - type_download (str): The type of data download ('csv', 'storage', 'bigquery').
            - data_path (str): The path to the data file or resource.

        Returns:
            - pd.DataFrame: The ingested data as a DataFrame.
    """
    try:
        if type_download == 'csv':
            data = pd.read_csv(data_path)
        elif type_download == 'storage':
            print("This path has been commented since I don't owned the Client API for any of these services.")
            # data = download_file_from_blob("bucket-name-in-project", "path-to-blob","filename.csv")
            data = None
        elif type_download == 'bigquery':
            print("This path has been commented since I don't owned the Client API for any of these services.")
            # data = ingest_bigquery_data()
            data = None
        return data
    except Exception as error:
        raise error