import requests
import json

from PIL import Image
from io import BytesIO

def download(pano_id: str, **kwargs):
    content =  requests.get('https://streetviewpixels-pa.googleapis.com/v1/tile', params={'cb_client': 'apiv3', 'panoid': pano_id, **kwargs}).content 
    return Image.open(BytesIO(content))

with open('../data/locs.json', 'r') as f:
    data = json.loads(f.read())['customCoordinates']

for i, loc in enumerate(data[:100]):
    print(f"Downloading {i}")

    img = download(loc['panoId'])
    img.save(f"../data/imgs/{loc['panoId']}.jpg")
