import pandas as pd

def print_metrics(csv_path):
    print(f"Reading metrics from {csv_path}...\n")
    df = pd.read_csv(csv_path)
    
    # Clean up column names in case there are hidden spaces
    df.columns = df.columns.str.strip()
    
    # Get the row from the final epoch
    final_epoch = df.iloc[-1]
    
    # ------------------ Bounding Box Metrics ------------------
    p_box = final_epoch['metrics/precision(B)']
    r_box = final_epoch['metrics/recall(B)']
    map50_box = final_epoch['metrics/mAP50(B)']
    map50_95_box = final_epoch['metrics/mAP50-95(B)'] # YOLO uses mAP50-95, acting as your 'mAP90' equivalent
    
    # Calculate F1 Score for bounding box
    f1_box = 2 * (p_box * r_box) / (p_box + r_box) if (p_box + r_box) > 0 else 0
    
    # ------------------ Segmentation Mask Metrics ------------------
    p_mask = final_epoch['metrics/precision(M)']
    r_mask = final_epoch['metrics/recall(M)']
    map50_mask = final_epoch['metrics/mAP50(M)']
    map50_95_mask = final_epoch['metrics/mAP50-95(M)']
    
    # Calculate F1 Score for mask
    f1_mask = 2 * (p_mask * r_mask) / (p_mask + r_mask) if (p_mask + r_mask) > 0 else 0
    
    print("=== Bounding Box (Detection) Metrics ===")
    print(f"Precision: {p_box:.4f}")
    print(f"Recall:    {r_box:.4f}")
    print(f"F1 Score:  {f1_box:.4f}")
    print(f"mAP@50:    {map50_box:.4f}")
    print(f"mAP@50-95: {map50_95_box:.4f}")
    
    print("\n=== Segmentation Mask Metrics ===")
    print(f"Precision: {p_mask:.4f}")
    print(f"Recall:    {r_mask:.4f}")
    print(f"F1 Score:  {f1_mask:.4f}")
    print(f"mAP@50:    {map50_mask:.4f}")
    print(f"mAP@50-95: {map50_95_mask:.4f}")

if __name__ == "__main__":
    # Path to the results file from your yolo26 trained segment model
    csv_path = "results_seg/results.csv"
    print_metrics(csv_path)
