FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir transformers==5.19.0 datasets==5.1.0 numpy==2.5.3 huggingface-hub==1.33.0 boto3==1.43.112

COPY extract_shards.py .

CMD ["python", "extract_shards.py"]