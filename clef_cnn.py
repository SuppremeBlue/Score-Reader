import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

DATASET_DIR = "dataset"  # dataset/train/*, dataset/val/*

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CNN MODEL
# -----------------------------
class ClefCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv2(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv3(x)))  # 16 -> 8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=transform
)
val_ds = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=transform
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

class_names = train_ds.classes
print("Classes:", class_names)

# -----------------------------
# TRAINING
# -----------------------------
model = ClefCNN(num_classes=len(class_names)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {running_loss:.4f}")

torch.save(model.state_dict(), "clef_cnn.pth")
print("Model saved to clef_cnn.pth")

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def classify_patch(model, img_patch):
    """
    img_patch: BGR or grayscale OpenCV image
    """
    model.eval()
    with torch.no_grad():
        if len(img_patch.shape) == 3:
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)

        x = transform(img_patch).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        cls = torch.argmax(probs, dim=1).item()
        confidence = probs[0, cls].item()

        return class_names[cls], confidence

# -----------------------------
# TEST ON A SINGLE IMAGE PATCH
# -----------------------------
if __name__ == "__main__":
    model.load_state_dict(torch.load("clef_cnn.pth", map_location=DEVICE))

    test_img = cv2.imread("test_patch.png")  # cropped clef candidate
    label, conf = classify_patch(model, test_img)

    print(f"Prediction: {label} ({conf:.3f})")
