# 2. Download Datasets
import os
from roboflow import Roboflow

# Dataset 1: Original IIT Pallakkad
rf_1 = Roboflow(api_key="bkvGKws3U501v3L3HEGw")
project_1 = rf_1.workspace("iit-pallakkad").project("detect-indian-currency")
ds_1 = project_1.version(2).download("yolo11")

# Dataset 2: New Partial/Overlapping Dataset
rf_2 = Roboflow(api_key="868TaqkwHhPtIstulvnK")
project_2 = rf_2.workspace("omkar-patkar-fes59").project("indian-currency-notes")
ds_2 = project_2.version(4).download("yolo11")

# 3. ULTIMATE MERGE: Harmonization + Sanitization + Counting
import yaml
import shutil
from pathlib import Path
from collections import Counter

def merge_and_harmonize(ds1_path, ds2_path, output_dir):
    unified_names = [
        '100Rupee_note', '10Rupee_note', '20Rupee_note', '50Rupee_note', 
        '200Rupee_note', '500Rupee_note', '2000Rupee_note',
        '1Rupee_coin', '2Rupee_coin', '5Rupee_coin', '10Rupee_coin',
        '5Rupee_note', 'None'
    ]
    
    name_map_ds2 = {
        '10': '10Rupee_note', '100': '100Rupee_note', '20': '20Rupee_note',
        '200': '200Rupee_note', '2000': '2000Rupee_note', '5': '5Rupee_note',
        '50': '50Rupee_note', '500': '500Rupee_note', 'None': 'None'
    }

    # Build ID Mappings
    with open(f"{ds1_path}/data.yaml", 'r') as f: ds1_classes = yaml.safe_load(f)['names']
    ds1_map = {i: (unified_names.index('None') if n=='undefined' else unified_names.index(n)) 
               for i, n in enumerate(ds1_classes)}
    
    with open(f"{ds2_path}/data.yaml", 'r') as f: ds2_classes = yaml.safe_load(f)['names']
    ds2_map = {i: unified_names.index(name_map_ds2.get(n, n)) for i, n in enumerate(ds2_classes)}

    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    fixed_count = 0
    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
        
        for folder, mapping, prefix in [(ds1_path, ds1_map, "ds1_"), (ds2_path, ds2_map, "ds2_")]:
            img_path_list = list(Path(f"{folder}/{split}/images").glob("*"))
            for img in img_path_list:
                new_name = f"{prefix}{img.name}"
                shutil.copy(img, f"{output_dir}/{split}/images/{new_name}")
                lbl = Path(f"{folder}/{split}/labels/{img.stem}.txt")
                if lbl.exists():
                    with open(lbl, 'r') as f_in, open(f"{output_dir}/{split}/labels/{new_name.replace(img.suffix, '.txt')}", 'w') as f_out:
                        for line in f_in:
                            p = line.split()
                            p[0] = str(mapping[int(p[0])])
                            # --- COORDINATE SANITIZER (CLIP TO 0.0-1.0) ---
                            coords = [float(x) for x in p[1:]]
                            if any(c > 1.0 or c < 0.0 for c in coords): fixed_count += 1
                            sanitized = [max(0.0, min(1.0, x)) for x in coords]
                            p[1:] = [f"{x:.6f}" for x in sanitized]
                            f_out.write(" ".join(p) + "\n")

    # Create Final YAML
    final_data = {
        'train': os.path.abspath(f"{output_dir}/train/images"),
        'val': os.path.abspath(f"{output_dir}/valid/images"),
        'test': os.path.abspath(f"{output_dir}/test/images"),
        'nc': len(unified_names),
        'names': unified_names,
        'task': 'detect' 
    }
    with open(f"{output_dir}/data.yaml", 'w') as f: yaml.dump(final_data, f)
    
    # Final Count Output
    print("--- Final Merged Class Counts ---")
    c = Counter()
    for lbl_f in Path(output_dir).rglob("labels/*.txt"):
        with open(lbl_f, 'r') as f:
            for line in f: 
                if line.strip(): c[int(line.split()[0])] += 1
    for i, name in enumerate(unified_names): print(f"{name}: {c[i]}")
    
    print(f"\n✅ Success: Sanitized {fixed_count} out-of-bounds labels.")
    print(f"Dataset Merged Successfully into {output_dir}/")

# 4. Execute Merge
merge_and_harmonize(ds_1.location, ds_2.location, 'merged_final')


# 5. Verification: Selective Sample Plotting
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_samples(path, split, num=5):
    with open(f"{path}/data.yaml", 'r') as f: names = yaml.safe_load(f)['names']
    imgs = list(Path(f"{path}/{split}/images").glob("*"))
    samples = random.sample(imgs, min(len(imgs), num))
    
    plt.figure(figsize=(20, 10))
    for i, img_p in enumerate(samples):
        img = cv2.imread(str(img_p)); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        lbl_p = Path(str(img_p).replace('images', 'labels').replace(img_p.suffix, '.txt'))
        if lbl_p.exists():
            with open(lbl_p, 'r') as f:
                for line in f:
                    p = list(map(float, line.split()))
                    cid = int(p[0]); coords = p[1:]
                    if len(coords) == 4:
                        x, y, bw, bh = coords
                        cv2.rectangle(img, (int((x-bw/2)*w), int((y-bh/2)*h)), (int((x+bw/2)*w), int((y+bh/2)*h)), (0,255,0), 3)
                    else:
                        pts = np.array([[int(coords[j]*w), int(coords[j+1]*h)] for j in range(0, len(coords), 2)], np.int32)
                        cv2.polylines(img, [pts], True, (255,0,0), 3)
                    cv2.putText(img, names[cid], (int(coords[0]*w), int(coords[1]*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        plt.subplot(1, num, i+1); plt.imshow(img); plt.axis('off')
    plt.show()

plot_samples('merged_final', 'valid', num=5)

# 6. Final Training with Auto-Discovery
from ultralytics import YOLO
import glob


initial_weights = glob.glob('/kaggle/input/**/*.pt', recursive=True)
if not os.path.exists(initial_weights):
    print(f"⚠️ {initial_weights} not found. Using baseline yolo11m.pt.")
    initial_weights = 'yolo11m.pt'
else:
    print(f"✅ Loading local weights: {initial_weights}")
        
model = YOLO(initial_weights)

results = model.train(
    data='merged_dataset_final/data.yaml', 
    task='detect',        
    epochs=150,          
    patience=50,         
    imgsz=640,
    batch=16,
    device=0,
    cache=True,          
    exist_ok=True,
    project="ViewSense_Final_Run",
    lr0=0.01,
    lrf=0.01303,
    momentum=0.94434,
    weight_decay=0.001,
    warmup_epochs=3.19989,
    warmup_momentum=0.95,
    box=6.22186,
    cls=0.68203,
    dfl=1.1111,
    hsv_h=0.02865,
    hsv_s=0.47952,
    hsv_v=0.39394,
    degrees=0.00155,
    translate=0.10293,
    scale=0.51575,
    shear=0.0,
    perspective=0.0,
    flipud=1.0e-05,
    fliplr=0.40516,
    bgr=0.00164,
    mosaic=1.0,
    mixup=0.00617,
    cutmix=0.0112,
    copy_paste=0.0,
    close_mosaic=10,
    name='merged_v26_training'
)