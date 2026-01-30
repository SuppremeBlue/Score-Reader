import os
import cv2
import numpy as np
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
EPOCHS = 8
LR = 1e-3

DATASET_DIR = "dataset"
TEMPLATE = "bass_clef.png"
PAGE = "debussy_1.png"

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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# -----------------------------
# TEMPLATE MATCHING (extract clef patches)
# -----------------------------
def find_clefs(page, template, scales=np.linspace(0.6, 1.4, 30), threshold=0.75):
    page_p = preprocess(page)
    tpl_p = preprocess(template)

    detections = []
    for scale in scales:
        tpl = cv2.resize(tpl_p, None, fx=scale, fy=scale)
        h, w = tpl.shape
        if h > page_p.shape[0] or w > page_p.shape[1]:
            continue

        result = cv2.matchTemplate(page_p, tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            detections.append((pt[0], pt[1], w, h, result[pt[1], pt[0]]))

    return detections

# -----------------------------
# NMS
# -----------------------------
def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    w  = boxes[:,2]
    h  = boxes[:,3]
    scores = boxes[:,4]

    x2 = x1 + w
    y2 = y1 + h

    idxs = scores.argsort()[::-1]
    pick = []

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = w[idxs[1:]] * h[idxs[1:]]
        iou = inter / area

        idxs = idxs[1:][iou < overlapThresh]

    return boxes[pick]

# -----------------------------
# DATASET CREATION
# -----------------------------
def make_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    train_bass = os.path.join(DATASET_DIR, "train", "bass_clef")
    train_noise = os.path.join(DATASET_DIR, "train", "noise")
    val_bass   = os.path.join(DATASET_DIR, "val", "bass_clef")
    val_noise  = os.path.join(DATASET_DIR, "val", "noise")

    for p in [train_bass, train_noise, val_bass, val_noise]:
        os.makedirs(p, exist_ok=True)

    # --- Extract bass clefs ---
    page = cv2.imread(PAGE)
    template = cv2.imread(TEMPLATE)

    detections = find_clefs(page, template, threshold=0.75)
    detections = non_max_suppression(detections, overlapThresh=0.4)

    print("Detected clefs:", len(detections))

    # save clef patches
    for i, (x, y, w, h, s) in enumerate(detections):
        patch = page[y:y+h, x:x+w]
        patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(train_bass, f"clef_{i}.png"), patch)

    # --- Create noise patches ---
    h, w = page.shape[:2]
    for i in range(400):
        x = np.random.randint(0, w - IMG_SIZE)
        y = np.random.randint(0, h - IMG_SIZE)
        patch = page[y:y+IMG_SIZE, x:x+IMG_SIZE]
        cv2.imwrite(os.path.join(train_noise, f"noise_{i}.png"), patch)

    print("Dataset created. Now split manually into train/val.")

# -----------------------------
# TRAINING
# -----------------------------
def train():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ClefCNN(num_classes=len(train_ds.classes)).to(DEVICE)
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
# INFERENCE
# -----------------------------
def classify_patch(model, img_patch, class_names):
    model.eval()
    with torch.no_grad():
        if len(img_patch.shape) == 3:
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        x = transform(img_patch).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        cls = torch.argmax(probs, dim=1).item()
        confidence = probs[0, cls].item()
        return class_names[cls], confidence

# -----------------------------
# FIND CLEFS IN A PAGE (full pipeline)
# -----------------------------
def detect_in_page(page_path):
    model = ClefCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("clef_cnn.pth", map_location=DEVICE))

    page = cv2.imread(page_path)
    template = cv2.imread(TEMPLATE)

    candidates = find_clefs(page, template, threshold=0.6)
    candidates = non_max_suppression(candidates, overlapThresh=0.4)

    class_names = ["bass_clef", "noise"]

    out = page.copy()
    for x, y, w, h, s in candidates:
        patch = page[y:y+h, x:x+w]
        patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))

        label, conf = classify_patch(model, patch, class_names)
        if label == "bass_clef" and conf > 0.9:
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Detections", out)
    cv2.waitKey(0)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Step 1: create dataset (once)
    make_dataset()

    # Step 2: Train model (after dataset is ready)
    # train()

    # Step 3: Detect clefs in a page
    detect_in_page("debussy_1.png")
