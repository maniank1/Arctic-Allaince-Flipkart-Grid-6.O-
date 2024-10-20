import cv2
import pytesseract
import re
import time

# Connects pytesseract to the trained tesseract module
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def process_roi(roi):
    # Get OCR data from the ROI (Region of Interest)
    data = pytesseract.image_to_string(roi)  # Extract raw text from the ROI
    return data

def extract_pack_size(text):
    # Regex pattern to detect pack size, assuming formats like '100ml', '250g', etc.
    pack_size_pattern = r'\d+\s?(ml|g|kg|L|litres|oz|grams|milliliters|liters|mg)'
    
    # Find pack size using regex
    pack_size_match = re.search(pack_size_pattern, text, re.IGNORECASE)
    pack_size = pack_size_match.group(0) if pack_size_match else "Pack size not found"
    
    return pack_size

def start_video_capture():
    # Start video capture from the default camera
    video = cv2.VideoCapture(0)

    # Set video frame width and height
    video.set(3, 640)  # Width
    video.set(4, 480)  # Height

    while True:
        # Capture video frame-by-frame
        ret, frame = video.read()
        if not ret:
            break

        # Define the virtual bounding box (x, y, width, height)
        x, y, w, h = 100, 100, 400, 200  # Example coordinates and dimensions
        # Draw the green box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with the green virtual box
        cv2.imshow('Live Video - OCR', frame)

        # Print capturing status and wait for 5 seconds before capturing
        print("Capturing frame in 5 seconds...")
        time.sleep(5)

        # Extract the region of interest (ROI) inside the green box
        roi = frame[y:y + h, x:x + w]
        
        # Perform OCR on the ROI and extract text
        detected_text = process_roi(roi)
        print("Detected Text (Brand Details):", detected_text)  # Print all detected text (brand details)
        
        # Extract and display the pack size explicitly
        pack_size = extract_pack_size(detected_text)
        print("Pack Size:", pack_size)

        # Break after capturing and processing the frame
        break

    # Release video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

# Start the video capture process
start_video_capture()