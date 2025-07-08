import requests

image_path = "image.png"

#загружал на свою вдску
url = "http://5.129.193.70:8000/ocr"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())