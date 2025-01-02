from azure.storage.blob import BlobClient, ContentSettings, BlobServiceClient
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO


def upload_blob_from_memory(data, sas_url, content_type="image/png"):
    """
    Uploads a file to Azure Blob Storage using a SAS URL directly from memory.

    Args:
    - data: Bytes data to upload.
    - sas_url: The full SAS URL of the blob.
    - content_type: MIME type of the file.
    """
    blob_client = BlobClient.from_blob_url(sas_url)

    # Check to see if the sas token worked
    print(f"Blob URL: {blob_client.url}")

    # Use ContentSettings for the content type
    content_settings = ContentSettings(content_type=content_type)

    # Upload blob with correct content settings
    blob_client.upload_blob(
        data, blob_type="BlockBlob", content_settings=content_settings, overwrite=True
    )

    print(f"Uploaded to blob: {sas_url}")
    return blob_client.url


def yolo_detect(image, model):
    """
    Process the given image with the YOLO model, annotate predictions, and generate labels.

    Args:
    - image (PIL.Image): Input image as a PIL object.
    - model: YOLO model.

    Returns:
    - annotated_image (PIL.Image): Image with predictions annotated.
    - labels (list of str): List of prediction labels.
    """
    # Resize image to YOLO-compatible size
    resized_image = image.resize((640, 640), Image.LANCZOS)

    # Predict using YOLO model
    results = model(resized_image)
    predictions = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    # Annotate image
    annotated_image_np = results[0].plot()
    annotated_image = Image.fromarray(annotated_image_np)

    # Create labels
    labels = [
        f"{int(classes[i])} {x1} {y1} {x2} {y2}"
        for i, (x1, y1, x2, y2) in enumerate(predictions)
    ]

    return annotated_image, labels, resized_image, predictions


def crop_to_closest_roof(image, predictions):
    """
    Crops a PIL image to the bounding box closest to the center of the image based on given predictions.

    Args:
    - image (PIL.Image): The source image.
    - predictions (list of lists): List of bounding boxes in the format [x1, y1, x2, y2, ...].
      Additional fields like confidence or class label may also be present.

    Returns:
    - PIL.Image: Cropped image.
    """
    # Convert PIL Image to a NumPy array for processing
    cv_image = np.array(image)

    # Get image dimensions
    height, width = cv_image.shape[:2]  # Handle both grayscale and color images
    image_center = (width / 2, height / 2)

    # Calculate distances to the image center for all bounding boxes
    distances = []
    for prediction in predictions:
        # Extract only the bounding box coordinates
        print(prediction)
        x1, y1, x2, y2 = prediction[:4]
        box_center = ((x1 + x2) / 2, (y1 + y2) / 2)  # Center of the bounding box
        distance = np.sqrt(
            (box_center[0] - image_center[0]) ** 2
            + (box_center[1] - image_center[1]) ** 2
        )
        distances.append(
            (distance, (x1, y1, x2, y2))
        )  # Store distance and box coordinates

    # Find the bounding box with the smallest distance
    closest_box = min(distances, key=lambda x: x[0])[1]
    x1, y1, x2, y2 = map(int, closest_box)

    # Crop the image using PIL (note the coordinates order)
    cropped_image = image.crop((x1, y1, x2, y2))

    return cropped_image


def generate_chat_completion(prompt, cropped_image_path, openai_api_key):

    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": cropped_image_path},
                    },
                ],
            }
        ],
    )

    return response
