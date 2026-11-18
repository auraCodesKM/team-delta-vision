import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

assert torch.cuda.is_available(), "CropScan training needs a CUDA GPU"
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
data = datasets.ImageFolder("data/plantvillage", transform=tf)
train, val = random_split(data, [0.8, 0.2], generator=torch.Generator().manual_seed(0))
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=len(data.classes)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

for epoch in range(3):
    model.train()
    for x, y in DataLoader(train, batch_size=32, shuffle=True):
        loss = model(pixel_values=x.cuda(), labels=y.cuda()).loss
        loss.backward()
        opt.step()
        opt.zero_grad()

model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in DataLoader(val, batch_size=64):
        pred = model(pixel_values=x.cuda()).logits.argmax(-1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
print(f"val accuracy {correct / total:.3f}")
torch.save(model.state_dict(), "cropscan.pt")
