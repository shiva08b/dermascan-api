FROM python:3.11-slim

WORKDIR /app

# Install system deps for opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download models from HuggingFace at build time
RUN python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('models', exist_ok=True)
files = ['screener.onnx', 'screener.onnx.data', 'classifier.onnx', 'classifier.onnx.data']
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(
        repo_id='shivanibutolia/dermascan-models',
        filename=f,
        local_dir='models',
        token=os.environ.get('HF_API_KEY')
    )
    print(f'✅ {f} done')
print('All models ready!')
"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]