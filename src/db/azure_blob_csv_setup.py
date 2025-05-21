from azure.storage.blob import BlobClient
from io import StringIO
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
base_sas_url = os.getenv("BLOB_BASE_URL")
blob_sas_token = os.getenv("BLOB_SAS_KEY")
# Define the SAS URL
sas_url = f"{base_sas_url}table/searchs.csv?{blob_sas_token}"
# Connect to the Blob using the SAS URL
blob_client = BlobClient.from_blob_url(sas_url)

#see if the sas token worked
print(f"Blob URL: {blob_client.url}")


# Create a DataFrame with the specified columns
data = {
    'address': [],
    'map_image_url': [],
    'annotated_image_url': [],
    'cropped_image_url': [],
    'labels': [],
    'response': [],
    'roof_detected': [],
    'timestamp': [],  # Add the timestamp column
    'ip_address': []  # Add the ip_address column
}

df = pd.DataFrame(data)

csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)

# Upload the CSV to Blob Storage
try:
    # Upload the contents of the buffer to the blob
    blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
except Exception as e:
    print(f"Failed to upload CSV: {e}")
