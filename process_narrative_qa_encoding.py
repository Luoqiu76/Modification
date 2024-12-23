
from bs4 import UnicodeDammit
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio
DIR = r"/data1/ytshi/narrativeqa/tmp"
async def process_encoding(file_path):
    with open(file_path, "rb") as f:
        encoding = UnicodeDammit(f.read()).original_encoding
    with open(file_path, "r", encoding=encoding) as f:
        text = f.read()
    text.encode("utf-8")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
tasks = []
for path in os.listdir(DIR):
    if path.endswith(".content"):
        tasks.append(process_encoding(os.path.join(DIR, path)))
loop = asyncio.get_event_loop()
loop.run_until_complete(tqdm_asyncio.gather(*tasks, desc="Processing encoding"))
loop.close()

for path in os.listdir(DIR):
    if path.endswith(".content"):
        with open(os.path.join(DIR, path), "r", encoding="utf-8") as f:
            f.read()
            pass