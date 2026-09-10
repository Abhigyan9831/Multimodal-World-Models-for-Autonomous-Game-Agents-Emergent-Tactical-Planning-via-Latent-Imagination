# perception/clip_encoder.py

import torch
import open_clip
import numpy as np
from PIL import Image

class CLIPEncoder:
    def __init__(self, device="cuda"):
        self.device = device

        print("[CLIP] Loading ViT-B/32 in fp16")

        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )

        # Move to GPU in fp16
        self.model = self.model.to(device=self.device, dtype=torch.float16)

        # Freeze all weights — no gradients, saves VRAM
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        print("[CLIP] Ready. Weights frozen. fp16 active.")

    @torch.no_grad()
    def encode(self, frame_numpy: np.ndarray) -> np.ndarray:
        """
        Input:  numpy array shape (64, 64, 3), dtype uint8, RGB
        Output: numpy array shape (1, 512), dtype float16
        """
        # Convert numpy → PIL → CLIP preprocess → tensor
        image = Image.fromarray(frame_numpy.astype(np.uint8))
        tensor = self.preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)
        tensor = tensor.to(device=self.device, dtype=torch.float16)

        # Encode — output is (1, 512)
        embedding = self.model.encode_image(tensor)

        return embedding.cpu().numpy()  # return as numpy for easy use


# ─── VALIDATION SCRIPT ──────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    encoder = CLIPEncoder(device="cuda")

    # Simulate a 64x64 RGB frame (random noise, like from UE5)
    fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    embedding = encoder.encode(fake_frame)

    print(f"\n Embedding shape: {embedding.shape}")   # expect (1, 512)
    print(f" Embedding dtype: {embedding.dtype}")     # expect float16
    print(f"VRAM after load:  {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # OOM check
    assert embedding.shape == (1, 512), " Shape wrong!"
    assert str(embedding.dtype) == "float16", " Not fp16!"
    print("\n Phase 2 validation PASSED. CLIP encoder is working.")