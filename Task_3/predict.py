import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = CNN().to(device)
model.load_state_dict(torch.load("Task_3/cnn_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

image_path = "image_path"
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

# ----------------------------
# EXTRA OUTPUT INFORMATION
# ----------------------------

print("\n=== PREDICTION RESULT ===")
print(f"Predicted class: {class_names[predicted.item()]}")
print(f"Confidence: {confidence.item() * 100:.2f}%")

print("\n=== TOP-3 PREDICTIONS ===")
top3_prob, top3_idx = torch.topk(probabilities, 3)
for i in range(3):
    print(f"{i+1}. {class_names[top3_idx[0][i].item()]}: {top3_prob[0][i].item() * 100:.2f}%")

print("\n=== RAW MODEL OUTPUT (Logits) ===")
print(output)

print("\n=== PROBABILITY FOR EACH CLASS ===")
for idx, cls in enumerate(class_names):
    print(f"{cls}: {probabilities[0][idx].item() * 100:.2f}%")
