# CropScan (Team Delta)

**Problem.** Smallholder farmers lose crops to leaf diseases they can't identify early.

**Solution.** Photograph a leaf: a fine-tuned Vision Transformer classifies the disease, then GPT-4o vision writes three practical advice points.

## Architecture
```
train.py   ViT (google/vit-base-patch16-224) fine-tuned on PlantVillage, 80/20 train/validation split,
           reports validation accuracy, saves cropscan.pt
advise.py  photo -> GPT-4o vision -> 3 advice bullet points
```

## Claims
- Validation accuracy **94.1%** on a held-out 20% split (computed in `train.py`; weights not included: 330 MB).
- Works on ordinary phone photos.

## Requirements
- NVIDIA GPU with 24 GB VRAM (CUDA 12): `train.py` refuses to run without one.
- `OPENAI_API_KEY` for the advice step.
- PlantVillage dataset (2.1 GB, not included).
- No hosted demo: inference needs the GPU.

## Run locally
```bash
pip install -r requirements.txt
python train.py                      # needs the GPU and data/plantvillage
OPENAI_API_KEY=... python advise.py leaf.jpg
```

---
*Synthetic benchmark submission, built to test an AI-hackathon evaluation pipeline. Not a real team's project.*
