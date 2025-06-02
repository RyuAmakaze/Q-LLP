# Use a PyTorch image with CUDA support so GPU acceleration is available.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# PyTorch is preinstalled in the base image. Install the remaining
# dependencies required for this project.
RUN pip install --no-cache-dir \
    qiskit==2.0.2 \
    qiskit_aer==0.17.0 \
    pytest==8.3.5 \
    pylatexenc

# Default command runs the training script.
CMD ["python", "src/run.py"]
