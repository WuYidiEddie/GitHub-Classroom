import torch
from torchvision import datasets, transforms
from transformers import ResNetForImageClassification

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

correct = 0
total = 0
for i in range(200):
    img, label = test_set[i]
    with torch.no_grad():
        logits = model(pixel_values=img.unsqueeze(0)).logits
    pred = logits.argmax(-1).item()
    if pred % 10 == label:
        correct += 1
    total += 1

acc = correct / total
print(f"accuracy={acc:.4f}")