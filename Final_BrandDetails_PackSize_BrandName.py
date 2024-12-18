import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re

# Load TrOCR model for OCR
processor = TrOCRProcessor.from_pretrained("models/trocr/pack_size")
model = VisionEncoderDecoderModel.from_pretrained("models/trocr/pack_size")

def extract_pack_size(image):
    """Extract pack size using OCR."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def process_pack_size(text):
    """Extract pack size (like 1kg, 500ml) using regex."""
    size = re.search(r"(\d+\s*(kg|g|ml|L))", text, re.IGNORECASE)
    return size.group(1) if size else "Pack Size Not Found"

def process_frame(frame):
    """Detect and annotate pack size."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = extract_pack_size(image)
    pack_size = process_pack_size(text)
    cv2.putText(frame, f"Pack Size: {pack_size}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Real-time video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_frame(frame)
    cv2.imshow("Pack Size Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()