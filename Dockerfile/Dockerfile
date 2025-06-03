# Use a PyTorch image with CUDA support so GPU acceleration is available.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies needed for building qiskit-aer with CUDA
RUN apt-get update && apt-get install -y \
    git build-essential cmake libopenblas-dev \
    && apt-get clean

# Install required Python packages (except qiskit-aer, we'll build that manually)
RUN pip install --no-cache-dir \
    qiskit==2.0.2 \
    pytest==8.3.5 \
    matplotlib \
    torchvision==0.16.0 \
    tqdm \
    pylatexenc \
    xformers

# Clone and build qiskit-aer with CUDA support
RUN git clone --branch stable/0.17 https://github.com/Qiskit/qiskit-aer.git && \
    cd qiskit-aer && \
    python setup.py bdist_wheel --define USE_CUDA=1 && \
    pip install dist/*.whl && \
    cd .. && rm -rf qiskit-aer

# Copy your project code
COPY . /app

# Default command
CMD ["python", "src/run.py"]
