import os
import yaml
from pathlib import Path
from collections import Counter
import shutil

def count_classes(dataset_path, dataset_name):
    with open(os.path.join(dataset_path, "data.yaml"), 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    
    counts = Counter()
    label_files = list(Path(dataset_path).rglob("labels/*.txt"))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] += 1
    
    print(f"\n--- {dataset_name} Counts ---")
    named_counts = {}
    for cid, count in counts.items():
        name = class_names[cid]
        named_counts[name] = count
        print(f"{name}: {count}")
    return named_counts

def merge_and_verify(ds1_path, ds2_path, output_dir):
    # 1. Defined Unified Classes
    # We prioritize DS1 names as they are more descriptive
    unified_names = [
        '100Rupee_note', '10Rupee_note', '20Rupee_note', '50Rupee_note', 
        '200Rupee_note', '500Rupee_note', '2000Rupee_note',
        '1Rupee_coin', '2Rupee_coin', '5Rupee_coin', '10Rupee_coin',
        '5Rupee_note', # New if exists in DS2
        'None' # Merged from undefined/None
    ]
    
    # Mapping for DS1
    with open(os.path.join(ds1_path, "data.yaml"), 'r') as f:
        ds1_classes = yaml.safe_load(f)['names']
    
    ds1_map = {}
    for i, name in enumerate(ds1_classes):
        if name == 'undefined':
            ds1_map[i] = unified_names.index('None')
        elif name in unified_names:
            ds1_map[i] = unified_names.index(name)
        else:
            # Fallback for any unexpected DS1 classes
            if name not in unified_names: unified_names.append(name)
            ds1_map[i] = unified_names.index(name)

    # Mapping for DS2
    with open(os.path.join(ds2_path, "data.yaml"), 'r') as f:
        ds2_classes = yaml.safe_load(f)['names']
        
    ds2_map = {}
    name_map_ds2 = {
        '10': '10Rupee_note', '100': '100Rupee_note', '20': '20Rupee_note',
        '200': '200Rupee_note', '2000': '2000Rupee_note', '5': '5Rupee_note',
        '50': '50Rupee_note', '500': '500Rupee_note', 'None': 'None'
    }
    
    for i, name in enumerate(ds2_classes):
        target_name = name_map_ds2.get(name, name)
        if target_name not in unified_names:
            unified_names.append(target_name)
        ds2_map[i] = unified_names.index(target_name)

    print(f"\nUnified Class List: {unified_names}")

    # 2. Merge Files
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)
        
        # Merge DS1
        for img in Path(f"{ds1_path}/{split}/images").glob("*"):
            new_name = f"ds1_{img.name}"
            shutil.copy(img, f"{output_dir}/{split}/images/{new_name}")
            label = Path(f"{ds1_path}/{split}/labels/{img.stem}.txt")
            if label.exists():
                with open(label, 'r') as f_in, open(f"{output_dir}/{split}/labels/{new_name.replace(img.suffix, '.txt')}", 'w') as f_out:
                    for line in f_in:
                        p = line.split()
                        p[0] = str(ds1_map[int(p[0])])
                        f_out.write(" ".join(p) + "\n")

        # Merge DS2
        for img in Path(f"{ds2_path}/{split}/images").glob("*"):
            new_name = f"ds2_{img.name}"
            shutil.copy(img, f"{output_dir}/{split}/images/{new_name}")
            label = Path(f"{ds2_path}/{split}/labels/{img.stem}.txt")
            if label.exists():
                with open(label, 'r') as f_in, open(f"{output_dir}/{split}/labels/{new_name.replace(img.suffix, '.txt')}", 'w') as f_out:
                    for line in f_in:
                        p = line.split()
                        p[0] = str(ds2_map[int(p[0])])
                        f_out.write(" ".join(p) + "\n")

    # 3. Create YAML
    with open(f"{output_dir}/data.yaml", 'w') as f:
        yaml.dump({'train': './train/images', 'val': './valid/images', 'test': './test/images', 'nc': len(unified_names), 'names': unified_names}, f)

    # 4. Final Verification
    print("\n--- Final Verification (Merged Dataset) ---")
    merged_counts = count_classes(output_dir, "Merged Dataset")
    return merged_counts

if __name__ == "__main__":
    d1_p = "dataset_1_temp"
    d2_p = "dataset_2_temp"
    
    print("Step 1: Counting original datasets...")
    c1 = count_classes(d1_p, "Dataset 1")
    c2 = count_classes(d2_p, "Dataset 2")
    
    print("\nStep 2 & 3: Merging and Verifying...")
    merged = merge_and_verify(d1_p, d2_p, "merged_dataset_final")
