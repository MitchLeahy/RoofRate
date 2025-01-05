
# import requests
# import base64



# class Yolov8RoboflowModel:
#     '''
#     YOLOv8 model for detecting buildings in satellite images.
#     Args:
#     - api_key (str): API key for Roboflow.
#     - model_id (str): ID of the hosted model on Roboflow.
#     - model_version (str): Version of the hosted model on (defaults to none which triggers a request to retreive the latest model version)
    
#     '''
#     def __init__(self, api_key: str, workspace_id: str , project_id: str , model_version: str = None):

        
#         self.api_key = api_key
#         self.workspace_id = workspace_id
#         self.project_id = project_id
#         self.roboflow_project = self.get_roboflow_project()
#         self.model_version = model_version
#         #gets the lattest model version if not speccifed this may need to be altered since this just graps the last object in the versions list
#         self.model_endpoint=  self.roboflow_project['versions'][-1]['model']['endpoint']
#         if self.model_version:
#             self.model_endpoint = self.model_endpoint.rsplit('/', 1)[0] +'/' + self.model_version
        


#     # def predict(self, image: Image) -> List[Detection]:
#     #     # Do prediction
#     #     return detections

#     def get_roboflow_project(self):
#         '''
#         Get the Roboflow project associated with the model.
#         '''
#         # Construct the URL
#         url = f"https://api.roboflow.com/{self.workspace_id}/{self.project_id}?api_key={self.api_key}"

#         # Make the GET request
#         response = requests.get(url)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             raise Exception(f"Error: {response.status_code}, {response.text}")
        
#         pass
#     # def get_latest_model_version(self, model_id: str):
#     def get_roof_detections(self, image_path: str):
#         '''
#         Predict the bounding boxes of buildings in the given image. need to change this to accept a pil image as opposed to base64?
#         Args:
#         - image_path (str): Path to the input image file.
#         Returns:
#         - predictions (dict): json Predictions object from the model.
#         '''
#         # Encode the image in base64
#         with open(image_path, "rb") as image_file:
#             image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

#         # Make the POST request
#         response = requests.post(
#             f"{self.model_endpoint}?api_key={self.api_key}",
#             data=image_base64,
#             headers={"Content-Type": "application/x-www-form-urlencoded"}
#         )

#         # Check the response
#         if response.status_code == 200:
#             return response.json()
#         else:
#             print("Error:", response.status_code, response.text)
