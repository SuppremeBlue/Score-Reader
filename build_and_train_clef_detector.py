import os, cv2, random, numpy as np
from glob import glob
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 64
AUG_PER_IMAGE = 5
NOISE_PATCHES_PER_PAGE = 200
TRAIN_VAL_SPLIT = 0.8

TEMPLATES_DIR = "templates"
PAGES_DIR = "pages"

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

CLEF_NAMES = ["bass", "treble", "alto", "tenor", "noise"]

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
# IMAGE PREPROCESS
# -----------------------------
def preprocess(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# -----------------------------
# STAFF LINE DETECTION
# -----------------------------
def detect_staff_lines(page):
    img = preprocess(page)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    horizontal_sum = np.sum(img == 0, axis=1)
    peaks = np.where(horizontal_sum > np.max(horizontal_sum) * 0.6)[0]
    return peaks


# -----------------------------
# TEMPLATE MATCHING
# -----------------------------
def find_clefs(page, template, threshold=0.75):
    page_p = preprocess(page)
    tpl_p = preprocess(template)
    if tpl_p.shape[0] > page_p.shape[0] or tpl_p.shape[1] > page_p.shape[1]:
        return []
    res = cv2.matchTemplate(page_p, tpl_p, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    h, w = tpl_p.shape
    candidates = []
    for pt in zip(*loc[::-1]):
        candidates.append((pt[0], pt[1], w, h, res[pt[1], pt[0]]))
    return candidates


# -----------------------------
# NMS
# -----------------------------
def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2], boxes[:, 3]
    scores = boxes[:, 4]

    x2, y2 = x1 + w, y1 + h
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
# CLASSIFY ALTO vs TENOR
# -----------------------------
def classify_alto_tenor(clef_y, clef_h, staff_lines):
    center_y = clef_y + clef_h // 2
    closest = staff_lines[np.argmin(np.abs(staff_lines - center_y))]
    idx = np.where(staff_lines == closest)[0][0] + 1  # 1-based

    if idx == 3:
        return "alto"
    elif idx == 4:
        return "tenor"
    else:
        return "unknown"


# -----------------------------
# AUGMENTATION
# -----------------------------
def augment(img):
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REPLICATE)

    factor = random.uniform(0.8, 1.2)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        noise = np.random.randn(IMG_SIZE, IMG_SIZE) * 8
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img


# -----------------------------
# SAVE PATCH
# -----------------------------
def save_patch(img, label, prefix, idx, train=True):
    base_dir = TRAIN_DIR if train else VAL_DIR
    out_dir = os.path.join(base_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{idx}.png")
    cv2.imwrite(path, img)


# -----------------------------
# DATASET BUILD
# -----------------------------
def build_dataset():
    templates = {
        "bass": cv2.imread(os.path.join(TEMPLATES_DIR, "bass.png")),
        "treble": cv2.imread(os.path.join(TEMPLATES_DIR, "treble.png")),
        "c_clef": cv2.imread(os.path.join(TEMPLATES_DIR, "c_clef.png")),
    }

    pages = glob(os.path.join(PAGES_DIR, "*.*"))

    # Create folders
    for d in [TRAIN_DIR, VAL_DIR]:
        for name in CLEF_NAMES:
            os.makedirs(os.path.join(d, name), exist_ok=True)

    counters = {name: 0 for name in CLEF_NAMES}

    for page_path in pages:
        page = cv2.imread(page_path)
        staff_lines = detect_staff_lines(page)

        # Detect bass and treble
        for name in ["bass", "treble"]:
            det = find_clefs(page, templates[name], threshold=0.75)
            det = non_max_suppression(det)
            for x, y, w, h, s in det:
                patch = cv2.resize(page[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                train = random.random() < TRAIN_VAL_SPLIT
                save_patch(patch, name, name, counters[name], train)
                counters[name] += 1

                for _ in range(AUG_PER_IMAGE):
                    save_patch(augment(patch), name, f"{name}_aug", counters[name], train)
                    counters[name] += 1

        # Detect C-clef and classify
        det = find_clefs(page, templates["c_clef"], threshold=0.72)
        det = non_max_suppression(det)
        for x, y, w, h, s in det:
            label = classify_alto_tenor(y, h, staff_lines)
            if label == "unknown":
                continue

            patch = cv2.resize(page[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            train = random.random() < TRAIN_VAL_SPLIT
            save_patch(patch, label, label, counters[label], train)
            counters[label] += 1

            for _ in range(AUG_PER_IMAGE):
                save_patch(augment(patch), label, f"{label}_aug", counters[label], train)
                counters[label] += 1

        # Noise patches
        h, w = page.shape[:2]
        for _ in range(NOISE_PATCHES_PER_PAGE):
            x = random.randint(0, w - IMG_SIZE)
            y = random.randint(0, h - IMG_SIZE)
            patch = cv2.cvtColor(page[y:y+IMG_SIZE, x:x+IMG_SIZE], cv2.COLOR_BGR2GRAY)
            train = random.random() < TRAIN_VAL_SPLIT
            save_patch(patch, "noise", "noise", counters["noise"], train)
            counters["noise"] += 1

    print("Dataset built:", counters)


# -----------------------------
# TRAIN
# -----------------------------
def train():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = ClefCNN(num_classes=len(train_ds.classes)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(8):
        model.train()
        loss_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        print(f"Epoch {epoch+1} loss: {loss_total:.4f}")

    torch.save(model.state_dict(), "clef_cnn.pth")
    print("Model saved.")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    build_dataset()
    train()
