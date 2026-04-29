# Run this as upload_models.py in your dermascan-api folder
import os

from huggingface_hub import HfApi, create_repo

# Your HF token should be provided via environment variable.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: Set HF_TOKEN in your environment before running this script.")

REPO_ID = "shivanibutolia/dermascan-models"  # change shiva08b to your HF username

api = HfApi()

# Create private repo
create_repo(REPO_ID, token=HF_TOKEN, private=True, repo_type="model")

# Upload all 4 model files
for filename in [
    "models/screener.onnx",
    "models/screener.onnx.data",
    "models/classifier.onnx",
    "models/classifier.onnx.data"
]:
    print(f"Uploading {filename}...")
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename.replace("models/", ""),
        repo_id=REPO_ID,
        token=HF_TOKEN,
        repo_type="model"
    )
    print(f"✅ {filename} uploaded")

print("🎉 All models uploaded to HuggingFace Hub!")
print(f"Repo: https://huggingface.co/{REPO_ID}")
