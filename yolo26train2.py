import roboflow
from ultralytics import YOLO
from roboflow import Roboflow
import yaml
import os
import glob

def main():
    print("--- Setting up YOLO26 Training ---")
   
    # rf = Roboflow(api_key="bkvGKws3U501v3L3HEGw")
    # project = rf.workspace("iit-pallakkad").project("detect-indian-currency")
    # version = project.version(2)
    # dataset = version.download("yolo26")
                
    print("\n--- Dataset Analysis ---")
    dataset_path = "Detect-Indian-Currency-2"
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"Num Classes: {data_config.get('nc', 'Unknown')}")
        print(f"Class Names: {data_config.get('names', 'Unknown')}")
        
    #     for split in ['train', 'valid', 'test']:
    #         img_dir = os.path.join(dataset_path, split, 'images')
    #         if os.path.exists(img_dir):
    #             count = len(glob.glob(os.path.join(img_dir, '*')))
    #             print(f"  - {split.ljust(6)} images: {count}")
    #         else:
    #             print(f"  - {split.ljust(6)} images: 0 (Directory not found)")
    # else:
    #     print(f"WARNING: data.yaml not found at {yaml_path}")

    # 3. Initialize Model
    model = YOLO("yolo26m.pt")

    # # 4. Hyperparameter Tuning (Find best settings)
    # print("\n--- Starting Hyperparameter Tuning ---")
    # best_hps = model.tune(
    #     data=yaml_path,
    #     epochs=10,        # Number of epochs per iteration
    #     iterations=8,    # Number of different hyperparameter combinations to try
    #     optimizer='AdamW',
    #     plots=True,       # Save tuning plots
    #     save=True,        # Save best hyperparameters
    #     val=True,
    #     devic
    # )
    # print("Hyperparameter Tuning Complete. Applying best parameters to final training...")

    # 5. Train Model with Tuned Parameters
    print("\n--- Starting YOLO26 Training with Tuned Hyperparameters ---")
    # best_hps["lr0"] = 0.003
    # We pass the best hyperparameters discovered during tuning to the train call
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        device=0,      
        batch=16,      
        exist_ok=True, 
        project="yolo26_currency_run",
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
        close_mosaic=10 # Apply the tuned hyperparameters here
    )
    print("Training Complete. Results saved to 'yolo26_currency_run'.")

if __name__ == '__main__':
    main()