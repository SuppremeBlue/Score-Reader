import os
import cv2
import numpy as np
import random
from glob import glob

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

CLEF_NAMES = ["bass", "treble", "alto", "tenor"]

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
# TEMPLATE MATCHING
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
# AUGMENTATION
# -----------------------------
def augment(img):
    angle = random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
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
# MAIN DATASET BUILD
# -----------------------------
def build_dataset():
    # load templates
    templates = {}
    for name in CLEF_NAMES:
        templates[name] = cv2.imread(os.path.join(TEMPLATES_DIR, f"{name}.png"))

    pages = glob(os.path.join(PAGES_DIR, "*.*"))

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    for d in [TRAIN_DIR, VAL_DIR]:
        for name in CLEF_NAMES:
            os.makedirs(os.path.join(d, name), exist_ok=True)
        os.makedirs(os.path.join(d, "noise"), exist_ok=True)

    counters = {name: 0 for name in CLEF_NAMES}
    counters["noise"] = 0

    for page_path in pages:
        page = cv2.imread(page_path)

        # ----- detect each clef -----
        for name, tpl in templates.items():
            detections = find_clefs(page, tpl, threshold=0.75)
            detections = non_max_suppression(detections, overlapThresh=0.4)

            for x, y, w, h, s in detections:
                patch = page[y:y+h, x:x+w]
                patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                train = random.random() < TRAIN_VAL_SPLIT
                save_patch(patch, name, name, counters[name], train=train)
                counters[name] += 1

                for _ in range(AUG_PER_IMAGE):
                    aug = augment(patch)
                    save_patch(aug, name, f"{name}_aug", counters[name], train=train)
                    counters[name] += 1

        # ----- noise patches -----
        h, w = page.shape[:2]
        for _ in range(NOISE_PATCHES_PER_PAGE):
            x = random.randint(0, w - IMG_SIZE)
            y = random.randint(0, h - IMG_SIZE)
            patch = page[y:y+IMG_SIZE, x:x+IMG_SIZE]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            train = random.random() < TRAIN_VAL_SPLIT
            save_patch(patch, "noise", "noise", counters["noise"], train=train)
            counters["noise"] += 1

    print("Dataset built!")
    print(counters)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    build_dataset()
