import cv2
import pytesseract
import re
import time

# Connects pytesseract to the trained tesseract module
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def process_roi(roi):
    # Get OCR data from the ROI (Region of Interest)
    data = pytesseract.image_to_string(roi)
    
    detected_text = data  # Store detected text from OCR
    return detected_text

def extract_mrp_and_expiry(text):
    # Pattern for MRP: Look for something like "MRP" followed by digits and possible currency symbols
    mrp_pattern = r"MRP[^\d]*(\d+\.?\d{0,2})"
    
    # Pattern for Expiry Date: Look for keywords like "Exp", "Expiry", followed by a date-like structure
    expiry_pattern = r"(Exp(?:iry)?\s*Date[:\s\-]*\d{2}\/\d{2}\/\d{4})"
    
    # Find MRP using regex
    mrp_match = re.search(mrp_pattern, text, re.IGNORECASE)
    mrp = mrp_match.group(1) if mrp_match else "MRP not found"
    
    # Find Expiry Date using regex
    expiry_match = re.search(expiry_pattern, text, re.IGNORECASE)
    expiry = expiry_match.group(0) if expiry_match else "Expiry date not found"
    
    return mrp, expiry

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

        # Capture key press
        key = cv2.waitKey(1)

        # After 5 seconds, automatically capture the frame
        print("Capturing frame in 5 seconds...")
        time.sleep(5)

        # Extract the region of interest (ROI) inside the green box
        roi = frame[y:y + h, x:x + w]
        detected_text = process_roi(roi)  # Perform OCR on the ROI
        
        # Extract MRP and Expiry Date using NLP (Regex)
        mrp, expiry = extract_mrp_and_expiry(detected_text)
        print(f"Detected MRP: {mrp}")
        print(f"Detected Expiry Date: {expiry}")

        # After capturing and processing, break the loop to exit
        break

    # Release video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

# Start the video capture process
start_video_capture()