from io import BytesIO
from PIL import Image 

import matplotlib.pyplot as plt
import numpy as np

import requests
import json
import cv2


def download(pano_id: str, **kwargs):
    content =  requests.get('https://streetviewpixels-pa.googleapis.com/v1/tile', params={'cb_client': 'apiv3', 'panoid': pano_id, **kwargs}).content 
    return Image.open(BytesIO(content))

with open('../data/locs.json', 'r') as f:
    data = json.loads(f.read())['customLocations']

loc = data[0]
img = download(loc['panoId'])


