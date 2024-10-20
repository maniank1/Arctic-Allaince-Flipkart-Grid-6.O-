import cv2
import pytesseract
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoProcessor
import time

# Configure Tesseract path (update this path according to your system)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Initialize Qwen2 model
def load_qwen2_model(model_path):
    """Load the fine-tuned Qwen2 Vision Transformer model."""
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()  # Put the model in evaluation mode
    return model, processor

# Load your fine-tuned Qwen2 model
MODEL_PATH = "C:/Users/harshit/Documents/models/qwen2/qwen2_fruit_model"  # Replace with your model path
qwen2_model, processor = load_qwen2_model(MODEL_PATH)

def detect_objects(image):
    """Perform object detection using the Qwen-2 Vision Transformer."""
    # Convert OpenCV BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Process image for Qwen2
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = qwen2_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    
    # Get predicted class name (replace this with your model's class mapping)
    class_names = qwen2_model.config.id2label
    predicted_class = class_names[predicted_class_idx]
    confidence = torch.softmax(logits, dim=-1)[0][predicted_class_idx].item()
    
    return predicted_class, confidence

def perform_ocr(image_path):
    """Perform OCR on the provided image using Tesseract."""
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Perform OCR using Tesseract
        ocr_text = pytesseract.image_to_string(image)
        
        if ocr_text.strip() == "":
            return "OCR Text: None detected"
        else:
            return f"OCR Text: {ocr_text}"
    except Exception as e:
        print(f"An error occurred during OCR: {str(e)}")
        return None

def analyze_image(image_path):
    """Analyze the image by performing OCR and object detection."""
    print(f"\nAnalyzing image: {image_path}")
    
    # Perform OCR on the image
    ocr_result = perform_ocr(image_path)
    print(ocr_result)
    
    # Save the analysis to a file
    with open("image_analysis.txt", "w") as file:
        file.write(ocr_result)
        
    return ocr_result

def capture_and_analyze_image(frame, x_start, y_start, x_end, y_end):
    print("\nCapturing image...")
    img_path = 'captured_image.jpg'
    cv2.imwrite(img_path, frame)
    print("Image captured and saved as:", img_path)

    # Crop the image using the bounding box coordinates
    cropped_image = frame[y_start:y_end, x_start:x_end]
    cropped_img_path = 'cropped_image.jpg'
    cv2.imwrite(cropped_img_path, cropped_image)

    # Perform object detection using Qwen2 Vision Transformer
    detected_class, confidence = detect_objects(cropped_image)
    
    # Analyze the cropped image (OCR)
    ocr_analysis = analyze_image(cropped_img_path)
    
    return detected_class, confidence, ocr_analysis

def capture_image_on_demand():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Define the static bounding box coordinates
    x_start, y_start = 100, 100
    x_end, y_end = 400, 300

    print("Press 'c' to capture an image manually. Images will be automatically captured every 7 seconds. Press 'q' to quit.")

    last_capture_time = time.time()
    capture_interval = 7

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get real-time object detection for the ROI
        roi = frame[y_start:y_end, x_start:x_end]
        detected_class, confidence = detect_objects(roi)

        # Draw the static bounding box on the frame
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        # Display detection results on frame
        text = f"{detected_class}: {confidence:.2f}"
        cv2.putText(frame, text, (x_start, y_start-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Webcam', frame)

        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            detected_class, confidence, analysis = capture_and_analyze_image(
                frame, x_start, y_start, x_end, y_end)
            print(f"\nDetected: {detected_class} (Confidence: {confidence:.2f})")
            last_capture_time = current_time

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            detected_class, confidence, analysis = capture_and_analyze_image(
                frame, x_start, y_start, x_end, y_end)
            print(f"\nDetected: {detected_class} (Confidence: {confidence:.2f})")
            last_capture_time = current_time
        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image_on_demand()
