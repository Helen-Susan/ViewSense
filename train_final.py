import torch
print(f"--- Hardware Check ---")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# STEP 2: Roboflow Downloads (All 6 Datasets)
import os
from roboflow import Roboflow

ROOT = os.getcwd()
DATASETS_DIR = os.path.join(ROOT, 'datasets')
os.makedirs(DATASETS_DIR, exist_ok=True)
os.chdir(DATASETS_DIR)

rf = Roboflow(api_key="VjWiwna9c7fEdneXcjw6")

print("Downloading Dataset 1... (ziptol)")
rf.workspace("tablet-ab3ji").project("ziptol").version(2).download("yolo26")

print("Downloading Dataset 2... (madhujr)")
rf.workspace("currency").project("2023-madhujr-reupload-mc").version(9).download("yolo26")

print("Downloading Dataset 3... (Coin detection)")
rf.workspace("anazs-workspace").project("indian-currency-coin-detection-97vpl").version(1).download("yolo26")

print("Downloading Dataset 4... (Detection FYXRZ)")
rf.workspace("anazs-workspace").project("currency-detection-fyxrz-y4iwi").version(1).download("yolo26")

print("Downloading Dataset 5... (Detection CGPJN)")
rf.workspace("anazs-workspace").project("currency-detection-cgpjn-ee3ez").version(2).download("yolo26")

print("Downloading Dataset 6... (Detect Indian Currency)")
rf.workspace("anazs-workspace").project("detect-indian-currency-3pmgy").version(1).download("yolo26")

os.chdir(ROOT)
print("✅ All Downloads Complete inside /datasets folder.")

# STEP 3: MASTER MERGE ENGINE (Harmonization + Sanitization)
import yaml
import shutil
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm

DATASETS_ROOT = os.path.join(os.getcwd(), 'datasets')
OUTPUT_DIR = os.path.join(os.getcwd(), 'merged_75k_dataset')

UNIFIED_NAMES = [
    '10Rupee_note', '20Rupee_note', '50Rupee_note', '100Rupee_note', 
    '200Rupee_note', '500Rupee_note', '2000Rupee_note', '5Rupee_note', 
    '1Rupee_coin', '2Rupee_coin', '5Rupee_coin', '10Rupee_coin', 'None'
]

MAPPINGS = {
    'ziptol-2': {'n10': '10Rupee_note', 'n20': '20Rupee_note', 'n50': '50Rupee_note', 'n100': '100Rupee_note', 'n200': '200Rupee_note', 'n500': '500Rupee_note'},
    '2023-madhujr-reupload-MC-9': {'n10': '10Rupee_note', 'n20': '20Rupee_note', 'n50': '50Rupee_note', 'n100': '100Rupee_note', 'n200': '200Rupee_note', 'n500': '500Rupee_note'},
    'Currency-detection-2': {
        '1- 10 Rupees': '10Rupee_note', '2- 20 Rupees': '20Rupee_note', '3- 50 Rupees': '50Rupee_note', 
        '4- 100 Rupees': '100Rupee_note', '5- 200 Rupees': '200Rupee_note', '6- 500 Rupees': '500Rupee_note', '7- 2000 Rupees': '2000Rupee_note'
    },
    'Currency-Detection-1': {
        '10 rupees': '10Rupee_note', '20 rupees': '20Rupee_note', '50 rupees': '50Rupee_note', 
        '100 rupees': '100Rupee_note', '200 rupees': '200Rupee_note', '500 rupees': '500Rupee_note'
    },
    'Detect-Indian-Currency-1': {
        '10Rupee_note': '10Rupee_note', '20Rupee_note': '20Rupee_note', '50Rupee_note': '50Rupee_note', 
        '100Rupee_note': '100Rupee_note', '200Rupee_note': '200Rupee_note', '500Rupee_note': '500Rupee_note', 
        '2000Rupee_note': '2000Rupee_note', '1Rupee_coin': '1Rupee_coin', '2Rupee_coin': '2Rupee_coin', 
        '5Rupee_coin': '5Rupee_coin', '10Rupee_coin': '10Rupee_coin', 'undefined': 'None'
    },
    'Indian-Currency-&-Coin-detection-1': {
        '10': '10Rupee_note', '20': '20Rupee_note', '50': '50Rupee_note', '100': '100Rupee_note', 
        '200': '200Rupee_note', '500': '500Rupee_note', '2000': '2000Rupee_note', 
        '1': '1Rupee_coin', '2': '2Rupee_coin', '5': '5Rupee_note', 
        '10_coin_ref': '10Rupee_coin'
    }
}

def merge_datasets():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    total_processed = 0
    sanitized_count = 0
    
    for ds_id, ds_name in enumerate(MAPPINGS.keys()):
        print(f"Processing {ds_name}...")
        ds_path = Path(DATASETS_ROOT) / ds_name
        yaml_path = ds_path / 'data.yaml'
        if not yaml_path.exists():
            print(f"  ⚠️ Skipping {ds_name} (data.yaml not found)")
            continue
        
        with open(yaml_path, 'r') as f: 
            orig_names = yaml.safe_load(f)['names']
            
        local_id_map = {}
        for i, old_name in enumerate(orig_names):
            target_name = MAPPINGS[ds_name].get(old_name, 'None')
            local_id_map[i] = UNIFIED_NAMES.index(target_name) if target_name in UNIFIED_NAMES else UNIFIED_NAMES.index('None')

        for split in ['train', 'valid', 'test']:
            src_img_dir = ds_path / split / 'images'
            if not src_img_dir.exists() and split == 'valid': src_img_dir = ds_path / 'val' / 'images'
            if not src_img_dir.exists(): continue
            
            src_lbl_dir = Path(str(src_img_dir).replace('images', 'labels'))
            img_files = list(src_img_dir.glob('*'))
            
            for img in tqdm(img_files, desc=f"  {split}", leave=False):
                new_name = f"ds{ds_id}_{img.name}"
                src_lbl = src_lbl_dir / (img.stem + ".txt")
                if not src_lbl.exists(): continue
                
                shutil.copy(img, Path(OUTPUT_DIR) / split / 'images' / new_name)
                with open(src_lbl, 'r') as f_in, open(Path(OUTPUT_DIR) / split / 'labels' / new_name.replace(img.suffix, '.txt'), 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if not parts: continue
                        new_cid = local_id_map.get(int(parts[0]), UNIFIED_NAMES.index('None'))
                        coords = []
                        for x in parts[1:]:
                            f_val = float(x)
                            if f_val < 0.0 or f_val > 1.0: sanitized_count += 1
                            coords.append(max(0.0, min(1.0, f_val)))
                        f_out.write(f"{new_cid} {' '.join([f'{c:.6f}' for c in coords])}\n")
                total_processed += 1

    yaml_data = {'train': os.path.join(OUTPUT_DIR, 'train', 'images'), 'val': os.path.join(OUTPUT_DIR, 'valid', 'images'), 'test': os.path.join(OUTPUT_DIR, 'test', 'images'), 'nc': len(UNIFIED_NAMES), 'names': UNIFIED_NAMES}
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f: yaml.dump(yaml_data, f)
    print(f"\n✅ Master Merge Complete. {total_processed} processed, {sanitized_count} sanitized.")

merge_datasets()
if os.path.exists(DATASETS_ROOT):
    shutil.rmtree(DATASETS_ROOT)
    print(f"✅ Deleted {DATASETS_ROOT} to free up space.")



from ultralytics import YOLO
import torch
import os
device_to_use = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {device_to_use}")
OUTPUT_DIR="merged_75k_dataset"
# Load previous best weights if available, else baseline
ROOT = os.getcwd()
initial_weights = 'yolo26m.pt'
    

model = YOLO(initial_weights)

# Start Training (Checkpoints are saved automatically in 'runs/' directory)
results = model.train(
    data=os.path.join(OUTPUT_DIR, 'data.yaml'),
    task='detect',        
    epochs=300,          
    patience=50,         
    imgsz=640,
    batch=16,
    device=device_to_use,
    cache=True,
    exist_ok=True,
    project="ViewSense_75k_Final",
    name='production_run',
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
    shear=0.0, perspective=0.0, flipud=1.0e-05, fliplr=0.40516, bgr=0.00164, mosaic=1.0, mixup=0.00617, cutmix=0.0112, copy_paste=0.0, close_mosaic=10
)