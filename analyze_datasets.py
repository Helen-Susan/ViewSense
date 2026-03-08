import os
import yaml
import sys
from pathlib import Path
from collections import Counter, defaultdict

DATASETS_ROOT = r'c:\Users\anasm\OneDrive\Documents\Projectss\Main-Project-SW\ViewSense\datasets'

mixed_datasets = [
    'ziptol-2',
    '2023-madhujr-reupload-MC-9'
]

normal_datasets = [
    'Currency detection.v1i.yolo26',
    'Currency-Detection-1',
    'Detect-Indian-Currency-1',
    'Indian-Currency-&-Coin-detection-1'
]

all_dataset_names = mixed_datasets + normal_datasets

def analyze_dataset(ds_name):
    print(f"Analyzing {ds_name}...", flush=True)
    ds_path = Path(DATASETS_ROOT) / ds_name
    yaml_path = ds_path / 'data.yaml'
    
    if not yaml_path.exists():
        print(f"  Error: {yaml_path} not found", flush=True)
        return None, None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data.get('names', [])
    stats = {
        'splits': defaultdict(int),
        'classes': defaultdict(lambda: Counter())
    }
    
    for split in ['train', 'valid', 'test']:
        labels_path = ds_path / split / 'labels'
        if not labels_path.exists():
            if split == 'valid':
                labels_path = ds_path / 'val' / 'labels'
            else:
                continue
        
        if not labels_path.exists(): continue
            
        print(f"  Scanning {split} split...", flush=True)
        try:
            label_files = list(labels_path.glob('*.txt'))
        except Exception as e:
            print(f"    Exception globbing {labels_path}: {e}", flush=True)
            continue
            
        stats['splits'][split] = len(label_files)
        print(f"    Found {len(label_files)} labels.", flush=True)
        
        for i, lf in enumerate(label_files):
            if i % 5000 == 0 and i > 0:
                print(f"    Processed {i} labels...", flush=True)
            try:
                with open(lf, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown_{cls_id}"
                            stats['classes'][split][cls_name] += 1
            except Exception as e:
                pass # skip corrupt files
    
    return stats, class_names

def print_stats(name, stats):
    print(f"\n========================================", flush=True)
    print(f" DATASET: {name}", flush=True)
    print(f"========================================\n", flush=True)
    
    total_imgs = sum(stats['splits'].values())
    print(f"TOTAL IMAGES: {total_imgs}", flush=True)
    for split, count in sorted(stats['splits'].items()):
        print(f"  - {split}: {count}", flush=True)
    
    print("\nClass Distribution (All Splits Combined):\n", flush=True)
    total_classes = Counter()
    for s_stats in stats['classes'].values():
        total_classes.update(s_stats)
    
    for cls, count in sorted(total_classes.items()):
        print(f"  - {cls}: {count}", flush=True)

try:
    results = {}
    for ds in all_dataset_names:
        stats, _ = analyze_dataset(ds)
        if stats:
            results[ds] = stats
            print_stats(ds, stats)
        else:
            print(f"Warning: Could not analyze {ds}", flush=True)

    mixed_agg = aggregate_group(mixed_datasets) if 'aggregate_group' in globals() else None
    
    def aggregate_group(ds_list):
        agg = {
            'splits': defaultdict(int),
            'classes': defaultdict(lambda: Counter())
        }
        for ds in ds_list:
            if ds in results:
                s = results[ds]
                for split, count in s['splits'].items():
                    agg['splits'][split] += count
                    for cls, cls_count in s['classes'][split].items():
                        agg['classes'][split][cls] += cls_count
        return agg

    mixed_agg = aggregate_group(mixed_datasets)
    normal_agg = aggregate_group(normal_datasets)

    print("\n\n" + "*"*60, flush=True)
    print(" GROUP AGGREGATION: MIXED/MULTIPLE NOTES (Handheld)", flush=True)
    print("*"*60, flush=True)
    print_stats("Mixed Aggregated", mixed_agg)

    print("\n\n" + "*"*60, flush=True)
    print(" GROUP AGGREGATION: NORMAL NOTES", flush=True)
    print("*"*60, flush=True)
    print_stats("Normal Aggregated", normal_agg)
except Exception as e:
    print(f"FATAL ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc(file=sys.stdout)
