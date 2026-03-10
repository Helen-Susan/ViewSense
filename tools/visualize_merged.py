import os
import cv2
import random
import yaml
import numpy as np
from pathlib import Path

def visualize_samples(dataset_path, split, num_samples=5, output_dir="previews"):
    # Load YAML for class names and colors
    with open(os.path.join(dataset_path, "data.yaml"), 'r') as f:
        data = yaml.safe_load(f)
    classes = data['names']
    
    # Create colors for each class
    random.seed(42)
    colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(classes))]
    
    img_dir = Path(dataset_path) / split / "images"
    lbl_dir = Path(dataset_path) / split / "labels"
    
    images = list(img_dir.glob("*"))
    samples = random.sample(images, min(len(images), num_samples))
    
    os.makedirs(f"{output_dir}/{split}", exist_ok=True)
    
    print(f"\n--- Visualizing {split} samples ---")
    
    for img_path in samples:
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    color = colors[cls_id]
                    label_text = f"{classes[cls_id]}"
                    
                    if len(coords) == 4: # Box: x, y, w, h
                        x_c, y_c, wb, hb = coords
                        x1 = int((x_c - wb/2) * w)
                        y1 = int((y_c - hb/2) * h)
                        x2 = int((x_c + wb/2) * w)
                        y2 = int((y_c + hb/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif len(coords) >= 6: # Polygon: x1, y1, x2, y2...
                        pts = []
                        for i in range(0, len(coords), 2):
                            pts.append([int(coords[i] * w), int(coords[i+1] * h)])
                        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, color, 2)
                        # Draw label at first point
                        cv2.putText(img, label_text, (pts[0][0][0], pts[0][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        save_path = f"{output_dir}/{split}/{img_path.name}"
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    dataset_path = "merged_dataset_final"
    visualize_samples(dataset_path, "train")
    visualize_samples(dataset_path, "valid")
    print("\nVisual verification complete! Check the 'previews' folder.")
