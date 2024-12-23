import os
import zipfile


DIR = r"/data1/ytshi/LVEval"

for filename in os.listdir(DIR):
    if filename.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(DIR, filename), 'r') as zip_ref:
            zip_ref.extractall(DIR)
            print(f"Extracted {filename}")