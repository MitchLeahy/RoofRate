import os
import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO, StringIO
import pandas as pd
from datetime import datetime
from azure.storage.blob import BlobClient
import roboflow
import openai
from src.app.utils import (
    draw_predictions,
    crop_to_closest_roof,
    generate_chat_completion,
    upload_blob_from_memory,
    create_yolov8_labels,
)

from dataclasses import dataclass
from PIL import Image

@dataclass
class RoofData:
    address: str
    map_image: Image.Image = None
    annotated_image: Image.Image = None
    cropped_image: Image.Image = None
    rating: str = ''
    roof_detected: bool = False
    map_image_sas_url: str = ''
    annotated_image_sas_url: str = ''
    cropped_image_sas_url: str = ''
    labels_sas_url: str = ''
    timestamp: str = ''
    ip_address: str = ''

class RoofImageProcessor:
    """
    Processes a single RoofData object:
      1) Fetches the map image
      2) Runs prediction
      3) Annotates/crops images, uploads them
      4) Generates a rating via OpenAI
    """

    # -- Setup Roboflow model as class-level or instance-level. 
    #    If you prefer to do it once, keep them as class variables below.
    load_dotenv()
    rf = roboflow.Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace().project("roofrate")
    versions = project.versions()
    VERSION_ID = versions[0].version
    model = project.version(VERSION_ID).model

    def __init__(self):
        self.base_sas_url = os.getenv("BLOB_BASE_URL")
        self.blob_sas_token = os.getenv("BLOB_SAS_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.maps_api_key = os.getenv("MAPS_API_KEY")

    def fetch_map_image(self, data: RoofData) -> None:
        """
        Fetches the map image for data.address,
        stores it in data.map_image,
        and uploads it to blob storage.
        """
        params = {
            "center": data.address,
            "zoom": "21",
            "size": "640x640",
            "maptype": "satellite",
            "key": self.maps_api_key,
        }
        response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params)
        if response.status_code == 200:
            # Store it in the data object
            data.map_image = Image.open(BytesIO(response.content))

            # Construct the SAS URL for storage
            data.map_image_sas_url = f"{self.base_sas_url}resized/{data.address}.png?{self.blob_sas_token}"

            # Upload to Blob
            map_image_io = BytesIO()
            data.map_image.save(map_image_io, format="PNG")
            map_image_io.seek(0)
            upload_blob_from_memory(
                map_image_io.getvalue(), 
                data.map_image_sas_url
            )
        else:
            raise ValueError(f"Failed to fetch map image for '{data.address}' (HTTP {response.status_code})")

    def run_prediction(self, data: RoofData) -> None:
        """
        Runs inference on the map_image_sas_url using Roboflow's hosted model,
        and stores the predictions in data.predictions.
        """
        if not data.map_image_sas_url:
            raise ValueError("Map image SAS URL is not set. Call fetch_map_image first.")
        
        results = self.model.predict(data.map_image_sas_url, hosted=True).json()
        data.predictions = results.get("predictions", [])

    def process_images(self, data: RoofData) -> None:
        """
        Creates annotated/cropped images, uploads them to blob storage,
        generates a rating via OpenAI. Also sets roof_detected appropriately.
        """
        if data.predictions:
            data.roof_detected = True

            # Create YOLO labels
            labels = create_yolov8_labels({"predictions": data.predictions})

            # Annotate & Crop
            data.annotated_image = draw_predictions(data.map_image, data.predictions)
            data.cropped_image = crop_to_closest_roof(data.map_image, data.predictions)

            # Generate SAS URLs
            data.annotated_image_sas_url = f"{self.base_sas_url}annotated/{data.address}.png?{self.blob_sas_token}"
            data.cropped_image_sas_url = f"{self.base_sas_url}cropped/{data.address}.png?{self.blob_sas_token}"
            data.labels_sas_url = f"{self.base_sas_url}labels/{data.address}.txt?{self.blob_sas_token}"

            # Upload annotated
            annotated_image_io = BytesIO()
            data.annotated_image.save(annotated_image_io, format="PNG")
            annotated_image_io.seek(0)
            upload_blob_from_memory(annotated_image_io.getvalue(), data.annotated_image_sas_url)

            # Upload cropped
            cropped_image_io = BytesIO()
            data.cropped_image.save(cropped_image_io, format="PNG")
            cropped_image_io.seek(0)
            upload_blob_from_memory(cropped_image_io.getvalue(), data.cropped_image_sas_url)

            # Upload labels
            labels_io = StringIO("\n".join(labels))
            upload_blob_from_memory(labels_io.getvalue().encode(), data.labels_sas_url, content_type="text/plain")

            # Generate rating with OpenAI
            with open("prompt.txt", "r") as prompt_file:
                prompt = prompt_file.read()

            response = generate_chat_completion(prompt, data.cropped_image_sas_url, self.openai_key)
            data.rating = response.choices[0].message.content

        else:
            data.rating = ""
            data.roof_detected = False

    def process_from_address(self, data: RoofData) -> RoofData:
        """
        Convenience method to call all steps in order.
        Updates the data object, returns it.
        """
        self.fetch_map_image(data)
        self.run_prediction(data)
        self.process_images(data)
        data.timestamp = datetime.now().isoformat()
        return data
class RoofBatchProcessor:
    """
    Processes multiple addresses using RoofImageProcessor.
    """

    def __init__(self):
        self.processor = RoofImageProcessor()

    def process(self, df: pd.DataFrame, address_column: str) -> pd.DataFrame:
        """
        For each address in the given column, creates a new RoofData,
        processes it, and returns a DataFrame of results.
        """
        results = []

        for address in df[address_column]:
            data = RoofData(address=address)
            try:
                # Process the property
                self.processor.process_from_address(data)

                # Convert the dataclass into a dictionary for easy tabular use
                results.append({
                    "address": data.address,
                    "map_image_sas_url": data.map_image_sas_url,
                    "annotated_image_sas_url": data.annotated_image_sas_url,
                    "cropped_image_sas_url": data.cropped_image_sas_url,
                    "labels_sas_url": data.labels_sas_url,
                    "rating": data.rating,
                    "roof_detected": data.roof_detected,
                    "timestamp": data.timestamp,
                    "ip_address": data.ip_address,
                    # add more fields if you need
                })
            except Exception as e:
                print(f"Failed to process {address}: {e}")
                # Optionally store partial/failure info in results
                results.append({
                    "address": address,
                    "error": str(e),
                    "roof_detected": False
                })

        return pd.DataFrame(results)
