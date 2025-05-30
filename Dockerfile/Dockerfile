# Base image with CUDA and `nvidia-smi` so GPU usage can be verified.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


# Install Python and pip. Create short aliases so `python`/`pip` work.
RUN apt-get update \ 
    && apt-get install -y python3 python3-pip \ 
    && ln -s /usr/bin/python3 /usr/local/bin/python \ 
    && ln -s /usr/bin/pip3 /usr/local/bin/pip \ 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support and the remaining dependencies.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 \ 
    && pip install --no-cache-dir qiskit qiskit_aer pytest

# Default command runs the training script.
CMD ["python", "src/run.py"]
