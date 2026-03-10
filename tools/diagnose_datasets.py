import os
import yaml
from roboflow import Roboflow
from pathlib import Path

def download_datasets():
    print("--- Downloading Dataset 1 ---")
    rf1 = Roboflow(api_key="bkvGKws3U501v3L3HEGw")
    project1 = rf1.workspace("iit-pallakkad").project("detect-indian-currency")
    version1 = project1.version(2)
    # Download as yolo11 to handle newer YOLO models
    ds1 = version1.download("yolo11", location="dataset_1_temp")

    print("\n--- Downloading Dataset 2 ---")
    rf2 = Roboflow(api_key="868TaqkwHhPtIstulvnK")
    project2 = rf2.workspace("omkar-patkar-fes59").project("indian-currency-notes")
    version2 = project2.version(4)
    ds2 = version2.download("yolo11", location="dataset_2_temp")
    
    return "dataset_1_temp", "dataset_2_temp"

def analyze_dataset(path, name):
    print(f"\n--- Analyzing {name} ---")
    data_yaml_path = os.path.join(path, "data.yaml")
    
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            print(f"Task type in YAML: {data.get('task', 'Not specified (defaulting to detect)')}")
            print(f"Number of classes: {data.get('nc')}")
            print(f"Classes: {data.get('names')}")
    
    issue_count = 0
    polygon_count = 0
    box_count = 0
    malformed_count = 0
    
    label_files = list(Path(path).rglob("labels/*.txt"))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                num_vals = len(parts)
                if num_vals == 5:
                    box_count += 1
                elif num_vals > 5:
                    polygon_count += 1
                    # A polygon needs at least 3 points, so 1 (class) + 2*3 = 7 values
                    if num_vals < 7:
                        print(f"Warning: Potential invalid polygon (too few points) in {label_file.name}: {line.strip()}")
                        issue_count += 1
                else:
                    # Less than 5 values is malformed for both box and polygon
                    print(f"Error: Malformed label line in {label_file.name}: {line.strip()}")
                    malformed_count += 1
                    issue_count += 1

    print(f"Summary for {name}:")
    print(f"- Total box labels: {box_count}")
    print(f"- Total polygon labels: {polygon_count}")
    print(f"- Malformed/Too short: {malformed_count}")
    print(f"- Potential Point issues: {issue_count}")
    
if __name__ == "__main__":
    d1, d2 = download_datasets()
    analyze_dataset(d1, "Dataset 1 (IIT Pallakkad)")
    analyze_dataset(d2, "Dataset 2 (Omkar Patkar)")
