"""Classify a leaf photo with the fine-tuned ViT (weights from train.py: cropscan.pt)."""

import sys

import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

LABELS = open("labels.txt").read().split()  # written by train.py, one class per line
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def classify(path: str) -> str:
    assert torch.cuda.is_available(), "CropScan inference needs a CUDA GPU"
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=len(LABELS))
    model.load_state_dict(torch.load("cropscan.pt"))
    model = model.cuda().eval()
    with torch.no_grad():
        logits = model(pixel_values=tf(Image.open(path).convert("RGB")).unsqueeze(0).cuda()).logits
    return LABELS[int(logits.argmax(-1))]


if __name__ == "__main__":
    print(classify(sys.argv[1]))
