from azure.storage.blob import BlobClient, ContentSettings, BlobServiceClient
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

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
    - predictions (list of dicts): List of predictions, where each dict contains:
        - x: Center x-coordinate of the bounding box.
        - y: Center y-coordinate of the bounding box.
        - width: Width of the bounding box.
        - height: Height of the bounding box.

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
        # Extract bounding box details
        box_center = (prediction["x"], prediction["y"])
        box_width = prediction["width"]
        box_height = prediction["height"]

        # Convert center coordinates and dimensions to corner coordinates
        x1 = box_center[0] - box_width / 2
        y1 = box_center[1] - box_height / 2
        x2 = box_center[0] + box_width / 2
        y2 = box_center[1] + box_height / 2

        # Calculate the distance of the bounding box center from the image center
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

def draw_predictions(image, predictions):
    """
    Draw bounding boxes and labels on an image.

    Args:
        image (PIL.Image.Image): The input image.
        predictions (list): A list of predictions, each with keys:
                            - x: center x-coordinate of the bounding box
                            - y: center y-coordinate of the bounding box
                            - width: width of the bounding box
                            - height: height of the bounding box
                            - class: label of the detected object
                            - confidence: confidence score of the detection

    Returns:
        PIL.Image.Image: The image with bounding boxes and labels drawn.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Default font

    for prediction in predictions:
        # Extract prediction details
        x, y = prediction["x"], prediction["y"]
        width, height = prediction["width"], prediction["height"]
        label = prediction["class"]
        confidence = prediction["confidence"]

        # Calculate bounding box coordinates
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2

        # Draw the bounding box
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # Draw the label and confidence score
        label_text = f"{label} ({confidence:.2f})"
        
        # Calculate text size using textbbox
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        # Draw a filled rectangle for the label background
        draw.rectangle([left, top - text_height - 4, left + text_width, top], fill="red")
        # Draw the label text
        draw.text((left, top - text_height - 2), label_text, fill="white", font=font)

    return image
def create_yolov8_labels(json_data):
    """
    Creates a YOLOv8 labels file from predictions in the JSON object.

    Args:
        json_data (dict): JSON object containing predictions and image metadata.
        output_path (str): Path to save the YOLOv8 labels file.

    Returns:
        None
    """
    # Extract predictions and image dimensions
    predictions = json_data["predictions"]
    image_width = 640
    image_height = 640

    # Prepare YOLOv8 label lines
    yolo_labels = []
    for prediction in predictions:
        class_id = prediction["class_id"]
        center_x = prediction["x"] / image_width
        center_y = prediction["y"] / image_height
        width = prediction["width"] / image_width
        height = prediction["height"] / image_height

        # Format: <class_id> <center_x> <center_y> <width> <height>
        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return yolo_labels