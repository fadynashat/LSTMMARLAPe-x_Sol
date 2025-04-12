import zipfile
import requests

# Download the full dataset (via direct URL)
url = "https://www.kaggle.com/api/v1/datasets/download/gauravdhamane/gwa-bitbrains"
response = requests.get(url, stream=True)

with open("bitbrains.zip", "wb") as f:
    for chunk in response.iter_content(chunk_size=128):
        f.write(chunk)

# Extract only the files you need (e.g., 1004.csv, 1005.csv)
with zipfile.ZipFile("bitbrains.zip", "r") as zip_ref:
    for file in ["1004.csv", "1005.csv"]:
        zip_ref.extract(file, path="bitbrains_data")