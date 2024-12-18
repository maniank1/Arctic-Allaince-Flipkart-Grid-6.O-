import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re

# Load pre-trained TrOCR model
processor = TrOCRProcessor.from_pretrained("models/trocr/mrp_expiry")
model = VisionEncoderDecoderModel.from_pretrained("models/trocr/mrp_expiry")

def extract_text(image):
    """Extract text using TrOCR."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def process_text(text):
    """Extract MRP and Expiry Date from text."""
    mrp = re.search(r"MRP[^\d]*(\d+\.\d+)", text, re.IGNORECASE)
    expiry = re.search(r"Expiry\s*Date[:\s]*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    return mrp.group(1) if mrp else "MRP Not Found", expiry.group(1) if expiry else "Expiry Not Found"

def process_frame(frame):
    """Process a single frame to detect MRP and expiry."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = extract_text(image)
    mrp, expiry = process_text(text)
    cv2.putText(frame, f"MRP: {mrp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Expiry: {expiry}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Real-time video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_frame(frame)
    cv2.imshow("MRP and Expiry Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()