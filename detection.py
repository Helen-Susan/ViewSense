from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def detect_objects(image_path, model_path="C:\\Users\\student\\View\\main-project-sw\\runs\\detect\\yolo26_currency_run\\train\\weights\\best.pt"):
    """
    Load an image and detect objects using the trained YOLO model.
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model weights
    
    Returns:
        results: YOLO detection results
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Make sure the model has been trained first by running yolo26train.py")
        return None
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running detection on: {image_path}")
    results = model.predict(image_path, conf=0.5)
    
    return results, image_path

def display_results(results, image_path):
    """
    Display detection results with bounding boxes.
    
    Args:
        results: YOLO detection results
        image_path (str): Path to the original image
    """
    
    if results is None or len(results) == 0:
        print("No detections found.")
        return
    
    # Get the first result
    result = results[0]
    
    # Print detection information
    print("\n--- Detection Results ---")
    print(f"Image: {image_path}")
    print(f"Number of detections: {len(result.boxes)}")
    
    if len(result.boxes) > 0:
        print("\nDetected Objects:")
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            confidence = box.conf[0].item()
            class_name = result.names[cls_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"  {i+1}. Class: {class_name} | Confidence: {confidence:.2f} | BBox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    else:
        print("No objects detected in the image.")
    
    # Display the image with detections
    annotated_image = result.plot()
    
    # Save the annotated image
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"\nAnnotated image saved to: {output_path}")
    
    # Display the image (optional, requires GUI)
    cv2.imshow("Detection Result", annotated_image)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """
    Main function to run detection on an image.
    """
    print("=== YOLO Currency Detection ===\n")
    
    # Get image path from user
    image_path = "C:\\Users\\student\\Downloads\\2ruppe.jpg"
    
    if not image_path:
        print("Error: No image path provided.")
        return
    
    # Run detection
    results = detect_objects(image_path)
    
    if results is not None:
        results, img_path = results
        display_results(results, img_path)

if __name__ == "__main__":
    main()
