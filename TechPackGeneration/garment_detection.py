import torch, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import SegformerImageProcessor,
AutoModelForSemanticSegmentation

# 1) Load & display the hoodie image
IMAGE_PATH = “imagepath”
pil_img = Image.open(IMAGE_PATH).convert("RGB")
image   = np.array(pil_img)
H, W    = image.shape[:2]

plt.figure(figsize=(6,6))
plt.imshow(pil_img)
plt.axis("off")
plt.title("Input Hoodie Art")
plt.show()

# 2) Initialize SegFormer-B2 (fashion parsing)
processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer-b2-fashion")
model     = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer-b2-fashion")
model.eval()

# 3) Run inference & upsample logits to full resolution
with torch.no_grad():
    inputs  = processor(images=pil_img, return_tensors="pt")
    outputs = model(**inputs)
logits = outputs.logits  # (1, C, H/4, W/4)

upsampled = torch.nn.functional.interpolate(
    logits,
    size=(H, W),
    mode="bilinear",
    align_corners=False
)
preds = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W) semantic map

# 4) Build raw masks for parts
hood   = (preds == 28)
sleeve = (preds == 32)
pocket = (preds == 33)
ys     = np.arange(H)[:, None]
cuff   = sleeve & (ys > H * 0.85)
body   = (preds != 0) & ~(hood | sleeve | pocket)

# 5) Split sleeve & cuff into left/right halves
xs            = np.arange(W)[None, :]
left_sleeve   = sleeve & (xs < W/2)
right_sleeve  = sleeve & (xs >= W/2)
left_cuff     = cuff   & (xs < W/2)
right_cuff    = cuff   & (xs >= W/2)

# 6) Helper: keep largest connected component & compute bbox
def largest_bbox(mask):
    comp_mask = mask.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(comp_mask)
    best_lbl, best_area = 0, 0
    for lbl in range(1, n_labels):
        area = (labels == lbl).sum()
        if area > best_area:
            best_area, best_lbl = area, lbl
    if best_area == 0:
        return None
    ys, xs = np.where(labels == best_lbl)
    return xs.min(), ys.min(), xs.max(), ys.max()

# 7) Compute bounding boxes for each part
parts = {
    "HOOD":        hood,
    "BODY":        body,
    "LEFT SLEEVE": left_sleeve,
    "RIGHT SLEEVE":right_sleeve,
    "LEFT CUFF":   left_cuff,
    "RIGHT CUFF":  right_cuff,
    "POCKET":      pocket,
}
part_boxes = {name: largest_bbox(mask) for name, mask in parts.items()}

# 8) Detect printed design by HSV color filtering
hsv         = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
lower_blue  = np.array([80, 50, 150])
upper_blue  = np.array([140, 255, 255])
mask_blue   = cv2.inRange(hsv, lower_blue, upper_blue)
kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask_blue   = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)

design_boxes = []
for c in contours:
    x, y, w_box, h_box = cv2.boundingRect(c)
    if w_box * h_box < 500:
        continue
    design_boxes.append((x, y, x + w_box, y + h_box))

# 9) Visualize all bounding boxes
fig, ax = plt.subplots(figsize=(12,8))
ax.imshow(image)
ax.axis("off")

# Draw part boxes in yellow
for name, box in part_boxes.items():
    if box is None:
        continue
    x0, y0, x1, y1 = box
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.text(x0, y0 - 5, name, color="red", fontsize=12, weight="bold",
va="bottom")

# Draw printed-design boxes in cyan dashed
for idx, (x0, y0, x1, y1) in enumerate(design_boxes, 1):
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             linewidth=2, edgecolor="black",
facecolor="none", linestyle="--")
    ax.add_patch(rect)
    ax.text(x0, y1 + 5, f"DESIGN {idx}", color="black", fontsize=12,
weight="bold", va="top")

# plt.title("Hoodie Parts (yellow) + Printed Design (cyan)")
plt.show()