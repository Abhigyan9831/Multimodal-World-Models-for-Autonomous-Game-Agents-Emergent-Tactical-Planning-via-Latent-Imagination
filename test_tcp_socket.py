from huggingface_hub import snapshot_download
import os

os.environ['HF_HOME'] = r'D:\HF_MODELS'

print("Starting LLaVA download (will resume if interrupted)")
snapshot_download(
    repo_id="llava-hf/llava-1.5-7b-hf",
    local_dir=r"D:\HF_MODELS\llava-1.5-7b",
    resume_download=True
)
print("Download complete")