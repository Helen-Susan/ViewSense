from ultralytics import YOLO
from roboflow import Roboflow
import yaml
import os
import glob

def main():
    print("--- Setting up YOLO26 Training ---")
    
    # 1. Download Dataset
    rf = Roboflow(api_key="868TaqkwHhPtIstulvnK")
    project = rf.workspace("anas-mohammed").project("detect-indian-currency-ldmie")
    version = project.version(1)
    print("Downloading dataset...")
    dataset = version.download("yolo26") 
    
    # 2. Analyze Dataset
    print("\n--- Dataset Analysis ---")
    dataset_path = dataset.location
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"Num Classes: {data_config.get('nc', 'Unknown')}")
        print(f"Class Names: {data_config.get('names', 'Unknown')}")
        
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(dataset_path, split, 'images')
            if os.path.exists(img_dir):
                count = len(glob.glob(os.path.join(img_dir, '*')))
                print(f"  - {split.ljust(6)} images: {count}")
            else:
                print(f"  - {split.ljust(6)} images: 0 (Directory not found)")
    else:
        print(f"WARNING: data.yaml not found at {yaml_path}")

    # 3. Train Model
    print("\n--- Starting YOLO26 Training on GPU ---")
    model = YOLO("yolo26m.pt") 
    
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        device=0,      
        batch=16,      
        exist_ok=True, 
        project="yolo26_currency_run" 
    )
    print("Training Complete. Results saved to 'yolo26_currency_run'.")

if __name__ == '__main__':
    main()
