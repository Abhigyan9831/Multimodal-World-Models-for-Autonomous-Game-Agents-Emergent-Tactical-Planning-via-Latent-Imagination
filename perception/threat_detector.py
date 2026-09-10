# perception/threat_detector.py
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import open_clip
import cv2

class ThreatDetector:
    def __init__(self, device='cuda'):
        self.device = device
        
        # YOLO for person detection
        print("[ThreatDetector] Loading YOLOv8n...")
        self.yolo = YOLO('yolov8n.pt')
        self.yolo.to(device)
        
        # CLIP for threat classification
        print("[ThreatDetector] Loading CLIP...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        self.clip_model = self.clip_model.to(device=device, dtype=torch.float16)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
            
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # Text embeddings for threat classification
        self.threat_texts = [
            "soldier holding rifle weapon gun",
            "person with assault rifle aiming",
            "armed military soldier with weapon"
        ]
        self.safe_texts = [
            "unarmed person standing",
            "person without weapon",
            "civilian without gun"
        ]
        
        # Pre-compute text embeddings
        with torch.no_grad(), torch.cuda.amp.autocast():
            threat_tokens = self.tokenizer(self.threat_texts).to(device)
            safe_tokens   = self.tokenizer(self.safe_texts).to(device)
            self.threat_embed = self.clip_model.encode_text(threat_tokens).mean(0)
            self.safe_embed   = self.clip_model.encode_text(safe_tokens).mean(0)
            self.threat_embed = self.threat_embed / self.threat_embed.norm()
            self.safe_embed   = self.safe_embed   / self.safe_embed.norm()

        print("[ThreatDetector] Ready ")

    @torch.no_grad()
    def classify_crop(self, crop_np):
        """Classify a cropped person image as threat or safe."""
        img   = Image.fromarray(crop_np.astype(np.uint8))
        tensor = self.clip_preprocess(img).unsqueeze(0).to(
            device=self.device, dtype=torch.float16
        )
        with torch.cuda.amp.autocast():
            img_embed = self.clip_model.encode_image(tensor)
            img_embed = img_embed / img_embed.norm()

        threat_score = (img_embed @ self.threat_embed.unsqueeze(-1)).item()
        safe_score   = (img_embed @ self.safe_embed.unsqueeze(-1)).item()

        is_threat = threat_score > safe_score
        confidence = torch.sigmoid(
            torch.tensor(threat_score - safe_score)
        ).item()
        return is_threat, confidence

    def detect(self, frame_np):
        """
        Full detection pipeline on one frame.
        
        Args:
            frame_np: numpy array (H, W, 3) RGB uint8
            
        Returns:
            detections: list of dicts with keys:
                box: (x1, y1, x2, y2)
                is_threat: bool
                confidence: float
                color: (R, G, B)
            threat_obs: numpy array (4,) = [norm_x, norm_y, threat_score, threat_detected]
            annotated_frame: numpy array with boxes drawn
        """
        H, W = frame_np.shape[:2]
        detections = []

        # ── YOLO detection ──────────────────────────────────────
        results = self.yolo(
            frame_np,
            classes=[0],      # class 0 = person
            conf=0.25,
            verbose=False
        )

        boxes = results[0].boxes
        threat_obs = np.zeros(4, dtype=np.float32)
        annotated  = frame_np.copy()

        if boxes is not None and len(boxes) > 0:
            best_threat_score = -1.0
            best_threat_box   = None

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Clamp to frame bounds
                x1 = max(0, min(x1, W-1))
                y1 = max(0, min(y1, H-1))
                x2 = max(x1+1, min(x2, W))
                y2 = max(y1+1, min(y2, H))

                crop = frame_np[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # ── CLIP classification ───────────────────────
                is_threat, confidence = self.classify_crop(crop)

                color = (255, 50, 50) if is_threat else (50, 255, 50)

                detections.append({
                    'box':        (x1, y1, x2, y2),
                    'is_threat':  is_threat,
                    'confidence': confidence,
                    'color':      color
                })

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"THREAT {confidence:.2f}" if is_threat else f"SAFE {confidence:.2f}"
                cv2.putText(
                    annotated, label, (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1, cv2.LINE_AA
                )

                # Track best threat for obs vector
                if is_threat and confidence > best_threat_score:
                    best_threat_score = confidence
                    best_threat_box   = (x1, y1, x2, y2)

            # Build threat observation vector
            if best_threat_box is not None:
                bx1, by1, bx2, by2 = best_threat_box
                cx = ((bx1 + bx2) / 2) / W  # normalized center x
                cy = ((by1 + by2) / 2) / H  # normalized center y
                threat_obs = np.array([cx, cy, best_threat_score, 1.0], dtype=np.float32)
            else:
                threat_obs = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        else:
            threat_obs = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        return detections, threat_obs, annotated


# ─── VALIDATION ─────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = ThreatDetector()

    # Test with fake frame
    fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    detections, threat_obs, annotated = detector.detect(fake_frame)

    print(f"Detections:  {len(detections)}")
    print(f"Threat obs:  {threat_obs}")
    print(f"Annotated shape: {annotated.shape}")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("\n ThreatDetector validation PASSED!")