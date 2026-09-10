# perception/vlm_encoder.py
import os
os.environ['HF_HOME'] = r'D:\HF_MODELS'
os.environ['TRANSFORMERS_CACHE'] = r'D:\HF_MODELS'

import torch
os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import open_clip

class VLMEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        
        print("[VLM] Loading LLaVA-1.5-7B (4-bit) from local disk...")
        local_model_path = r"D:\HF_MODELS\llava-1.5-7b"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            local_model_path,
            quantization_config=bnb_config,
            device_map="auto", 
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(local_model_path, use_fast=False)
        self.model.eval()
        
        print("[VLM] Loading CLIP Text Encoder...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.clip_model = self.clip_model.to(device).half().eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        print("[VLM] Ready")

    @torch.no_grad()
    def encode(self, frame_numpy: np.ndarray, ai_perception: np.ndarray = None):
        image = Image.fromarray(frame_numpy.astype(np.uint8)).convert("RGB")
    
        if ai_perception is not None and ai_perception[0] > 0:
            actors = int(ai_perception[0] * 5)
            dist   = ai_perception[1] * 5000
            prompt = (f"USER: <image>\nI am an NPC soldier. I can see {actors} enemy(s) "
                  f"approximately {dist:.0f} units away. "
                  f"Describe the tactical situation and best action briefly.\nASSISTANT:")
        else:
            prompt = "USER: <image>\nI am an NPC soldier. Describe this scene and any threats briefly.\nASSISTANT:"
    
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # FIX 2: Safely move inputs to the model's device without calling .to on the 4-bit model
        input_device = self.model.get_input_embeddings().weight.device
        inputs = {
            k: v.to(input_device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    
        output_ids = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = self.processor.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
        del inputs, output_ids
        torch.cuda.empty_cache()
    
        text_tokens = self.clip_tokenizer([generated_text]).to(self.device)
        with torch.cuda.amp.autocast():
            text_embed = self.clip_model.encode_text(text_tokens)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    
        return text_embed.cpu().numpy(), generated_text   

# ─── VALIDATION ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"VRAM before load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    vlm = VLMEncoder()
    
    # Test with a fake 64x64 frame
    fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    embed, text = vlm.encode(fake_frame)
    
    print(f"VLM Embedding shape: {embed.shape}")
    print(f"Embedding dtype: {embed.dtype}")
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    assert embed.shape == (1, 512), " Shape wrong!"
    #assert str(embed.dtype) == "float16", "Not fp16!"
    print("\n VLM Encoder validation PASSED!")