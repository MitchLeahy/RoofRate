import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from azure.storage.blob import BlobClient

# Load environment variables
load_dotenv()
base_sas_url = os.getenv("BLOB_BASE_URL")
blob_sas_token = os.getenv("BLOB_SAS_KEY")

# Change 'some_file.csv' to the actual CSV you want to display
csv_sas_url = f"{base_sas_url}table/searchs.csv?{blob_sas_token}"

st.title("View CSV from Azure Blob")

def load_csv_from_blob():
    try:
        blob_client = BlobClient.from_blob_url(csv_sas_url)
        downloaded_blob = blob_client.download_blob().readall()
        return pd.read_csv(StringIO(downloaded_blob.decode("utf-8")))
    except Exception as e:
        st.error(f"Failed to read CSV from Azure: {e}")
        return None

df = load_csv_from_blob()
if df is not None:
    st.dataframe(df)
    st.success("CSV Loaded Successfully")
    
