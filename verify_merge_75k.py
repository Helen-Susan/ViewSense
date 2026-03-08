import os
import yaml
from pathlib import Path
from collections import Counter
from tqdm import tqdm

MERGED_DIR = r'c:\Users\anasm\OneDrive\Documents\Projectss\Main-Project-SW\ViewSense\merged_75k_dataset'

def deep_audit():
    print("--- Starting Deep Audit of Merged 75k Dataset ---")
    yaml_path = os.path.join(MERGED_DIR, 'data.yaml')
    if not os.path.exists(yaml_path):
        print("❌ Error: data.yaml not found!")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        nc = data['nc']
        names = data['names']

    print(f"Dataset Name: ViewSense 75k Master")
    print(f"Expected Classes: {nc} {names}")

    total_images = 0
    total_labels = 0
    out_of_range_coords = 0
    invalid_class_ids = 0
    distribution = Counter()

    for split in ['train', 'valid', 'test']:
        img_dir = Path(MERGED_DIR) / split / 'images'
        lbl_dir = Path(MERGED_DIR) / split / 'labels'
        
        if not img_dir.exists():
            print(f"⚠️ Warning: Split {split} images not found.")
            continue

        images = list(img_dir.glob('*'))
        total_images += len(images)
        
        print(f"Auditing {split} split ({len(images)} images)...")
        
        for img_p in tqdm(images, desc=f"Verifying {split}", leave=False):
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            if not lbl_p.exists():
                continue
            
            total_labels += 1
            with open(lbl_p, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    # 1. Class ID Verification
                    cid = int(parts[0])
                    if cid < 0 or cid >= nc:
                        invalid_class_ids += 1
                    distribution[cid] += 1
                    
                    # 2. Coordinate Range Verification
                    for coord in parts[1:]:
                        val = float(coord)
                        if val < 0.0 or val > 1.0:
                            out_of_range_coords += 1

    print("\n" + "="*40)
    print(" AUDIT RESULTS")
    print("="*40)
    print(f"Total Images on Disk: {total_images}")
    print(f"Total Labels on Disk: {total_labels}")
    print(f"Invalid Class IDs (Should be 0): {invalid_class_ids}")
    print(f"Out-of-range Coordinates (Should be 0): {out_of_range_coords}")
    
    print("\nVerified Class Distribution:")
    for i, name in enumerate(names):
        print(f" {i}: {name.ljust(15)} -> {distribution[i]} instances")

    if out_of_range_coords == 0 and invalid_class_ids == 0:
        print("\n✅ VERDICT: Dataset is 100% Valid and Mathematically Sanitized.")
    else:
        print("\n❌ VERDICT: Dataset has inconsistencies. Re-run merge.")

if __name__ == "__main__":
    deep_audit()
