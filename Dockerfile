FROM python:3.10-slim

WORKDIR /app

# opencv-python-headless still dynamically links against these even without
# any GUI/display support - missing on a bare python:3.10-slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libxext6 libsm6 libxrender1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY scripts/infer.py scripts/infer.py

ENTRYPOINT ["python", "scripts/infer.py"]
