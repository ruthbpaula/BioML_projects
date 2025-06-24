import os
import json
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# === Dataset Loader for your .jsonl structure ===
class VacuoleDataset(Dataset):
    def __init__(self, jsonl_path, local_img_dir, transforms=None):
        self.transforms = transforms
        self.local_img_dir = local_img_dir
        with open(jsonl_path, 'r') as f:
            self.entries = [json.loads(line) for line in f]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        filename = os.path.basename(entry['image_url'])
        img_path = os.path.join(self.local_img_dir, filename)

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ Failed to load image: {img_path}\n{e}")
            return self.__getitem__((idx + 1) % len(self))

        # Build bounding boxes
        boxes = []
        labels = []
        for obj in entry['label']:
            try:
                x1 = obj['topX']
                y1 = obj['topY']
                x2 = x1 + obj['width']
                y2 = y1 + obj['height']
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # vacuole = class 1
            except KeyError:
                continue

        if len(boxes) == 0:
            print(f"⚠️ No labels for image: {img_path}, skipping.")
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.entries)

# === Model Setup ===
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# === Train ===
def train_model(jsonl_path, image_dir, output_model_path):
    dataset = VacuoleDataset(jsonl_path, image_dir, transforms=F.to_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])

    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    for epoch in range(5):
        total_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), output_model_path)
    print(f"✅ Model saved to {output_model_path}")

# === Inference & Visualization ===
def predict_and_count(model_path, image_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]['boxes'].cpu()
    scores = outputs[0]['scores'].cpu()
    count = sum(score > threshold for score in scores)

    print(f"🔎 Detected vacuoles: {count}")

    fig, ax = plt.subplots()
    ax.imshow(img)
    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1.5, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
    plt.title(f"Vacuoles Detected: {count}")
    plt.axis('off')
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    jsonl_file = "C:/Users/ruthb/Downloads/Azure_ML_studio_vacuoles_count/brain_vacuole_labels_grouped_azureml.jsonl"
    image_dir = "C:/Users/ruthb/Downloads/vacuoles_test_images"
    model_output = "C:/Users/ruthb/Downloads/Azure_ML_studio_vacuoles_count/vacuole_detector.pth"
    test_image = "C:/Users/ruthb/Downloads/Azure_ML_studio_vacuoles_count/F2.large_test.jpg"  # or any other test image

    train_model(jsonl_file, image_dir, model_output)
    predict_and_count(model_output, test_image)
