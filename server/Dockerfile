FROM python:3.11
WORKDIR /app


RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install numpy opencv-python-headless accelerate draccus uvicorn fastapi json-numpy

RUN pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
# Force CUDA 11.8 PyTorch instead of PyPI's default CUDA 13 wheel
RUN pip install --force-reinstall --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118
CMD ["python", "main.py"]