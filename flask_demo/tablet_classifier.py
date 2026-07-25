"""
Tablet-image classifier.

Wraps the ResNet18 model (model.pth) + class list (classes.json) that map a
photo of a pill/tablet to a medicine name, so the Flask interaction app can
offer "upload a photo" as an alternative to picking a drug name from a
dropdown.

Kept deliberately separate from app.py / inference.py: this model has
nothing to do with the DrugSafetyHGNN graph model, it just turns pixels
into a class-name string. app.py is responsible for resolving that string
into a graph CID.
"""
import io
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_TOP_K = 3


class TabletClassifier:
    """Loads once at startup, then classify() is called per uploaded image."""

    def __init__(self, model_path: str, classes_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes: List[str] = self._load_classes(classes_path)
        self.model = self._load_model(model_path, len(self.classes)) if self.classes else None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @property
    def ready(self) -> bool:
        return bool(self.classes) and self.model is not None

    @staticmethod
    def _load_classes(path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                classes = json.load(f)
            print(f"[classifier] Loaded {len(classes)} classes from '{path}'")
            return classes
        except Exception as e:
            print(f"[classifier] Error loading classes from '{path}': {e}")
            return []

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        # weights=None: the checkpoint below supplies every weight (including
        # a resized fc layer), so there's no need to also download ImageNet
        # pretrained weights just to immediately overwrite them.
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _load_model(self, model_path: str, num_classes: int):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model = self._build_model(num_classes)
            state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            print(f"[classifier] Loaded model weights from '{model_path}'")
            return model
        except Exception as e:
            print(f"[classifier] Error loading model weights from '{model_path}': {e}")
            return None

    @torch.no_grad()
    def classify(self, image_bytes: bytes, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Returns [{"class": <name>, "probability": <float>}, ...] sorted by probability desc."""
        if not self.ready:
            raise RuntimeError("Classifier model/classes failed to load.")

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idx = torch.topk(probs, k=min(top_k, len(self.classes)))

        return [
            {"class": self.classes[i], "probability": float(p)}
            for p, i in zip(top_probs, top_idx)
        ]