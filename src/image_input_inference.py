import os

import requests
from dotenv import load_dotenv

# Load the stored environment variables
load_dotenv()

# Get API_TOKEN
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

##################################################
# Image Classification
# https://huggingface.co/google/vit-base-patch16-224

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"


def image_classification_query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


output = image_classification_query("data/tiger.jpg")

##################################################
# Image Classification
# https://huggingface.co/facebook/detr-resnet-50-panoptic

API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50-panoptic"


def image_segementation_query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


output = image_segementation_query("data/dog-cat.jpg")

from PIL import Image

image = Image.open("data/dog-cat.jpg")
for element in output:
    print(element["label"])
    img_mask = element["mask"]
    Image.composite(image, image, mask=img_mask)
    image.show()

cat_mask = output[0]["mask"]
cat_mask = output[0]["mask"]
