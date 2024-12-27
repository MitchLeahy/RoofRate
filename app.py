import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import os
import requests
from src.app.utils import (
    predict,
    crop_to_closest_roof,
    generate_chat_completion,
    upload_blob_from_memory,
)
from ultralytics import YOLO
from io import BytesIO
from azure.storage.blob import BlobClient, ContentSettings, BlobServiceClient
import io

# Load environment variables
load_dotenv()
base_sas_url = os.getenv("BLOB_BASE_URL")
blob_sas_token = os.getenv("BLOB_SAS_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
maps_api_key = os.getenv("MAPS_API_KEY")

with open ("prompt.txt", "r") as prompt:
    prompt = prompt.read()

# Load your YOLO model
model = YOLO("src/model/best.pt")  # Replace with your YOLO model path

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
        map_image = Image.open(BytesIO(response.content))

        # Run the prediction
        st.write("Processing image with YOLO model...")
        annotated_image, labels, resized_image, predictions = predict(map_image, model)

       

        # Crop the image
        cropped_image = crop_to_closest_roof(map_image, predictions)

        # Show results
        col1, col2 = st.columns(2)

        # Display annotated image
        col1.image(annotated_image, caption="Annotated Image", use_container_width=True)
        # Display cropped image
        col2.image(cropped_image, caption="Cropped Image", use_container_width=True)
        
        cropped_image_sas_url = f"{base_sas_url}cropped/{address}.png?{blob_sas_token}"
        annotated_image_sas_url = (
            f"{base_sas_url}annotated/{address}.png?{blob_sas_token}"
        )
        labels_sas_url = f"{base_sas_url}labels/{address}.txt?{blob_sas_token}"
        resized_image_sas_url = f"{base_sas_url}resized/{address}.png?{blob_sas_token}"

        # Save and upload annotated image
        annotated_image_io = io.BytesIO()
        annotated_image.save(annotated_image_io, format="PNG")
        annotated_image_io.seek(0)
        annotated_image_path = upload_blob_from_memory(
            annotated_image_io.getvalue(), annotated_image_sas_url
        )

        # Save and upload resized image
        resized_image_io = io.BytesIO()
        resized_image.save(resized_image_io, format="PNG")
        resized_image_io.seek(0)
        resized_image_path = upload_blob_from_memory(
            resized_image_io.getvalue(), resized_image_sas_url
        )

        # Save and upload cropped image
        cropped_image_io = io.BytesIO()
        cropped_image.save(cropped_image_io, format="PNG")
        cropped_image_io.seek(0)
        cropped_image_path = upload_blob_from_memory(
            cropped_image_io.getvalue(), cropped_image_sas_url
        )

        # Save and upload labels
        labels_io = io.StringIO("\n".join(labels))
        labels_path = upload_blob_from_memory(
            labels_io.getvalue().encode(), labels_sas_url, content_type="text/plain"
        )


        response = generate_chat_completion(prompt, cropped_image_path, openai_key)

        # Parse and display the response
        string = response.choices[0].message.content
        st.title("Rating: ")
        st.header(string)

    else:
        st.error(
            f"Failed to fetch the map for the address. Error code: {response.status_code}"
        )
