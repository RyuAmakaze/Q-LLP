FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch torchvision qiskit pytest qiskit_aer
CMD ["python", "src/run.py"]
