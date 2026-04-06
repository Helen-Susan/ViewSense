import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = r"results_seg\results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Strip any leading/trailing whitespace from column names just in case
    df.columns = df.columns.str.strip()

    # Extract the epochs
    epochs = df['epoch']
    
    plt.figure(figsize=(10, 6))
    
    # Plot mAP50 for Box (B) and Mask (M) if they exist
    if 'metrics/mAP50(B)' in df.columns:
        plt.plot(epochs, df['metrics/mAP50(B)'], label='Bounding Box mAP-50', linewidth=2)
    
    if 'metrics/mAP50(M)' in df.columns:
        plt.plot(epochs, df['metrics/mAP50(M)'], label='Segmentation Mask mAP-50', linewidth=2)

    plt.title('mAP-50 vs. Epoch (up to 100 epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP-50')
    plt.xlim(0, 100)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Save the plot
    save_path = "map50_vs_epoch.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
