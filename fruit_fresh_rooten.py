import cv2
import torch
from transformers import AutoModelForImageClassification, AutoProcessor
import numpy as np
from PIL import Image
import time

class FruitFreshnessDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load your fine-tuned Qwen2 model and processor
        self.model = AutoModelForImageClassification.from_pretrained(model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
    def preprocess_image(self, image):
        # Convert OpenCV BGR to RGB
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Preprocess image using the Qwen2 processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def predict_freshness(self, image):
        inputs = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Assuming your model outputs freshness percentage
            freshness_score = torch.sigmoid(outputs.logits).item() * 100
            
        return freshness_score

    def determine_freshness_category(self, freshness_score):
        if freshness_score >= 80:
            return "Fresh", (0, 255, 0)  # Green
        elif freshness_score >= 50:
            return "Moderately Fresh", (0, 255, 255)  # Yellow
        else:
            return "Not Fresh", (0, 0, 255)  # Red

def capture_and_analyze():
    # Initialize the detector with your model path
    detector = FruitFreshnessDetector("C:/Users/harshit/Documents/models/qwen2/qwen2_fruit_model")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the bounding box coordinates
    x_start, y_start = 100, 100
    x_end, y_end = 400, 300

    print("Press 'c' to capture and analyze. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Draw bounding box
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'c' to analyze", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Fruit Freshness Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Crop the region of interest
            cropped = frame[y_start:y_end, x_start:x_end]
            
            # Analyze freshness
            freshness_score = detector.predict_freshness(cropped)
            category, color = detector.determine_freshness_category(freshness_score)
            
            # Display results
            result_frame = frame.copy()
            result_text = f"Freshness: {freshness_score:.1f}% - {category}"
            cv2.putText(result_frame, result_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Save the analyzed image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"analyzed_fruit_{timestamp}.jpg", result_frame)
            
            # Display the result for a few seconds
            cv2.imshow('Analysis Result', result_frame)
            cv2.waitKey(3000)  # Display for 3 seconds
            
        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

def analyze_single_image(image_path):
    detector = FruitFreshnessDetector("C:/Users/harshit/Documents/models/qwen2/qwen2_fruit_model")
    
    # Read and analyze the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return
    
    freshness_score = detector.predict_freshness(image)
    category, color = detector.determine_freshness_category(freshness_score)
    
    # Display results
    result_text = f"Freshness: {freshness_score:.1f}% - {category}"
    cv2.putText(image, result_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Analysis Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Capture and analyze from webcam")
    print("2. Analyze a single image")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        capture_and_analyze()
    elif choice == "2":
        image_path = input("Enter the path to your image: ")
        analyze_single_image(image_path)
    else:
        print("Invalid choice. Exiting.")