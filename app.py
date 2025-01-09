import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import os
import requests
from io import StringIO
from src.app.utils import (
    draw_predictions,
    crop_to_closest_roof,
    generate_chat_completion,
    upload_blob_from_memory,
    create_yolov8_labels,
)

from io import BytesIO
from azure.storage.blob import BlobClient, ContentSettings, BlobServiceClient
import io
import roboflow
from datetime import datetime
import pandas as pd

# Load environment variables
load_dotenv()
base_sas_url = os.getenv("BLOB_BASE_URL")
blob_sas_token = os.getenv("BLOB_SAS_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
maps_api_key = os.getenv("MAPS_API_KEY")
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

roof_table_sas_url = f"{base_sas_url}table/searchs.csv?{blob_sas_token}"
map_image_path = ''
annotated_image_path = ''
cropped_image_path = ''
labels_path = ''

blob_client = BlobClient.from_blob_url(roof_table_sas_url)
try:
    # Read the blob's content
    downloaded_blob = blob_client.download_blob().readall()
    # Load it into a Pandas DataFrame
    roof_df = pd.read_csv(StringIO(downloaded_blob.decode("utf-8")))
except Exception as e:
    raise ValueError(f"Failed to read CSV: {e}")



with open ("prompt.txt", "r") as prompt:
    prompt = prompt.read()


# Load your YOLO model
# model = YOLO("src/model/best.pt")  # Replace with your YOLO model path

# Streamlit app title
st.title("Roof Rate")


# Address input
address = st.text_input("Enter the address:")

if address:
    # Static map URL
    static_map_url = "https://maps.googleapis.com/maps/api/staticmap"

    # Define parameters for the Google Maps API
    params = {
        "center": address,
        "zoom": "21",
        "size": "640x640",
        "maptype": "satellite",
        "key": maps_api_key,
    }

    # Fetch the static map image
    response = requests.get(static_map_url, params=params)


    if response.status_code == 200:
        # Open the map image using PIL
        map_image_sas_url = f"{base_sas_url}resized/{address}.png?{blob_sas_token}"
        cropped_image_sas_url = f"{base_sas_url}cropped/{address}.png?{blob_sas_token}"
        annotated_image_sas_url = (
            f"{base_sas_url}annotated/{address}.png?{blob_sas_token}"
        )
        labels_sas_url = f"{base_sas_url}labels/{address}.txt?{blob_sas_token}"

        map_image = Image.open(BytesIO(response.content))
        map_image_io = io.BytesIO()
        map_image.save(map_image_io, format="PNG")
        map_image_io.seek(0)
        map_image_path = upload_blob_from_memory(
            map_image_io.getvalue(), map_image_sas_url
        )

        # Run the prediction
        st.write("Processing image with YOLO model...")
        
        rf = roboflow.Roboflow(api_key=roboflow_api_key)
        project = rf.workspace().project("roofrate")
        versions = project.versions()
        VERSION_ID = versions[0].version

        model = project.version(VERSION_ID).model

        results = model.predict(map_image_path, hosted=True).json()
        if results['predictions'] != []:
            labels = create_yolov8_labels(results)
            predictions = results["predictions"]
            annotated_image = draw_predictions(map_image, predictions)


            # annotated_image, labels, resized_image, predictions = predict(map_image, model)

        

            # # Crop the image
            cropped_image = crop_to_closest_roof(map_image, predictions)

            # # Show results
            col1, col2 = st.columns(2)

            # # Display annotated image
            col1.image(annotated_image, caption="Annotated Image", use_container_width=True)
            # # Display cropped image
            col2.image(cropped_image, caption="Cropped Image", use_container_width=True)
        
            
            

            # Save and upload annotated image
            annotated_image_io = io.BytesIO()
            annotated_image.save(annotated_image_io, format="PNG")
            annotated_image_io.seek(0)
            annotated_image_path = upload_blob_from_memory(
                annotated_image_io.getvalue(), annotated_image_sas_url
            )

            # Save and upload resized image
        

            # Save and upload cropped image
            cropped_image_io = io.BytesIO()
            cropped_image.save(cropped_image_io, format="PNG")
            cropped_image_io.seek(0)
            cropped_image_path = upload_blob_from_memory(
                cropped_image_io.getvalue(), cropped_image_sas_url
            )
            labels_io = io.StringIO("\n".join(labels))
            labels_path = upload_blob_from_memory(
                labels_io.getvalue().encode(), labels_sas_url, content_type="text/plain"
            )



            response = generate_chat_completion(prompt, cropped_image_path, openai_key)

            # # Parse and display the response
            string = response.choices[0].message.content
            st.title("Rating: ")
            st.header(string)
            new_data_row = {
                "address": address,
                "map_image_url": map_image_sas_url,
                "annotated_image_url": annotated_image_sas_url,
                "cropped_image_url": cropped_image_sas_url,
                "labels": labels_sas_url,
                "response": string,
                "roof_detected": True,
                "timestamp": datetime.now().isoformat(),
                "ip_address": '',
            }
            roof_df = pd.concat([roof_df,pd.DataFrame([new_data_row])], ignore_index=True)
            st.dataframe(roof_df)
            csv_buffer = StringIO()
            roof_df.to_csv(csv_buffer, index=False)


        else:
            st.image(map_image, caption="Map Image", use_container_width=True)
            st.error("No roofs detected in the image.")
            new_data_row = {
                "address": address,
                "map_image_url": map_image_sas_url,
                "annotated_image_url": annotated_image_sas_url,
                "cropped_image_url": cropped_image_sas_url,
                "labels": labels_sas_url,
                "response": '',
                "roof_detected": False,
                "timestamp": datetime.now().isoformat(),
                "ip_address": '',
            }
            roof_df = pd.concat([roof_df,pd.DataFrame([new_data_row])], ignore_index=True)
            st.dataframe(roof_df)
            csv_buffer = StringIO()
            roof_df.to_csv(csv_buffer, index=False)

    else:
        st.error(
            f"Failed to fetch the map for the address. Error code: {response.status_code}"
        )
