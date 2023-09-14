import sys

from google.cloud import storage

bucket = sys.argv[1]
prefix = sys.argv[2]
client = storage.Client()
b = client.get_bucket(bucket)

blobs = b.list_blobs(prefix=prefix)
seen = set()
for blob in blobs:
    prefix = blob.name.rsplit("/", 1)[0]
    dir = f"gs://{b.name}/{prefix}"
    if dir not in seen:
        print(dir)
        seen.add(dir)
