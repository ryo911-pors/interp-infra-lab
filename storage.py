
import os
import uuid
import io
import boto3
import numpy as np

s3 = boto3.client("s3")

def split_s3(path):
    bucket, key = path[len("s3://"):].split("/", 1)
    return bucket, key

def write_bytes(path, data):
    if path.startswith("s3://"):
        bucket, key = split_s3(path)
        s3.put_object(Bucket=bucket, Key=key, Body=data)
    else:
        tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

def read_bytes(path):
    if path.startswith("s3://"):
        bucket, key = split_s3(path)
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def load_npy(path):
    return np.load(io.BytesIO(read_bytes(path)))


def exists(path):
    if path.startswith("s3://"):
        bucket, key = split_s3(path)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError:
            return False

    return os.path.exists(path)

def join(a, b):                   
    if a.startswith("s3://"):
        return a.rstrip("/") + "/" + b
    return os.path.join(a, b)

def save_npy(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    write_bytes(path, buf.getvalue())

    