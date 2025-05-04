import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from PIL import Image
import requests
import streamlit.components.v1 as components

from azure.storage.blob import BlobClient
from datetime import datetime
import uuid

# Import your classes
from src.app.roof_processing.processing import RoofImageProcessor, RoofData

def show():
    # Load environment variables
    load_dotenv()
    base_sas_url = os.getenv("BLOB_BASE_URL")
    blob_sas_token = os.getenv("BLOB_SAS_KEY")
    roof_table_sas_url = f"{base_sas_url}table/searchs.csv?{blob_sas_token}"
    maps_api_key = os.getenv("MAPS_API_KEY")

    # Read the existing CSV from Blob
    blob_client = BlobClient.from_blob_url(roof_table_sas_url)
    try:
        downloaded_blob = blob_client.download_blob().readall()
        roof_df = pd.read_csv(StringIO(downloaded_blob.decode("utf-8")))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Streamlit app title
    st.title("Roof Rate - Analysis")

    # Create a simpler approach - just use Streamlit's text input
    address = st.text_input("Enter address:", 
                           help="Type an address and select from the dropdown suggestions")
    
    # Process button
    if st.button("Process Address") and address:
        
        # 1. Create a dataclass instance
        data = RoofData(address=address)

        # 2. Create a processor and run the full pipeline
        processor = RoofImageProcessor()
        try:
            processor.process_from_address(data)
        except Exception as e:
            st.error(f"Failed to process address: {e}")
            st.stop()

        # 3. Check if a roof was detected
        if data.roof_detected:
            # Show annotated & cropped images side-by-side
            col1, col2 = st.columns(2)
            if data.annotated_image:
                col1.image(
                    data.annotated_image,
                    caption="Annotated Image",
                    use_container_width=True  # Updated
                )
            if data.cropped_image:
                col2.image(
                    data.cropped_image,
                    caption="Cropped Image",
                    use_container_width=True  # Updated
                )

            # Show rating
            st.title("Rating:")
            st.header(data.rating)

            new_data_row = {
                "address": data.address,
                "map_image_url": data.map_image_sas_url,
                "annotated_image_url": data.annotated_image_sas_url,
                "cropped_image_url": data.cropped_image_sas_url,
                "labels": data.labels_sas_url,
                "response": data.rating,
                "roof_detected": data.roof_detected,
                "timestamp": data.timestamp,
                "ip_address": data.ip_address,
            }
            roof_df = pd.concat([roof_df, pd.DataFrame([new_data_row])], ignore_index=True)
            
        else:
            # If no roof was detected
            st.image(data.map_image, caption="Map Image", use_container_width=True)  # Updated
            st.error("No roofs detected in the image.")

            new_data_row = {
                "address": data.address,
                "map_image_url": data.map_image_sas_url,
                "annotated_image_url": data.annotated_image_sas_url,
                "cropped_image_url": data.cropped_image_sas_url,
                "labels": data.labels_sas_url,
                "response": data.rating,
                "roof_detected": data.roof_detected,
                "timestamp": data.timestamp,
                "ip_address": data.ip_address,
            }
            roof_df = pd.concat([roof_df, pd.DataFrame([new_data_row])], ignore_index=True)
            

        # 4. Upload updated CSV to Blob Storage
        csv_buffer = StringIO()
        roof_df.to_csv(csv_buffer, index=False)
        try:
            blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
            st.success("Results saved successfully!")
        except Exception as e:
            st.error(f"Failed to upload CSV: {e}")

# # This allows the file to be run directly as a standalone page
# if __name__ == "__main__":
#     show()
