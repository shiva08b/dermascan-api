from huggingface_hub import hf_hub_download
import os

os.makedirs('models', exist_ok=True)

files = [
    'screener.onnx',
    'screener.onnx.data', 
    'classifier.onnx',
    'classifier.onnx.data'
]

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