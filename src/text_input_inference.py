import os

import requests
from dotenv import load_dotenv

# Load the stored environment variables
load_dotenv()

# Get API_TOKEN
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# TextClassification
# https://huggingface.co/ProsusAI/finbert?text=Stocks+rallied+and+the+British+pound+gained.
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"


def text_classification_query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = text_classification_query({
    "inputs": "I like you. I love you",
})

##################################################
# Text Generation
# https://huggingface.co/meta-llama/Llama-2-7b
# Only with Meta license

# https://huggingface.co/defog/sqlcoder
# No API yet, but better as gpt3.5-turbo and downloadable

# https://huggingface.co/Deci/DeciCoder-1b
# Decoder only
API_URL = "https://api-inference.huggingface.co/models/Deci/DeciCoder-1b"


def text_generation_query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = text_generation_query({
    "inputs": "What is Physics?",
})

##################################################
# https://huggingface.co/google/pegasus-large


API_URL = "https://api-inference.huggingface.co/models/google/pegasus-large"


def summarization_query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output = summarization_query({
    "inputs": "The tower is 324 metres (1,063 ft) tall, about the same height "
              "as an 81-storey building, and the tallest structure in Paris. "
              "Its base is square, measuring 125 metres (410 ft) on each side. "
              "During its construction, the Eiffel Tower surpassed the Washington "
              "Monument to become the tallest man-made structure in the world, "
              "a title it held for 41 years until the Chrysler Building in "
              "New York City was finished in 1930. It was the first structure to "
              "reach a height of 300 metres. Due to the addition of a broadcasting "
              "aerial at the top of the tower in 1957, it is now taller than the "
              "Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, "
              "the Eiffel Tower is the second tallest free-standing structure "
              "in France after the Millau Viaduct.",
})

