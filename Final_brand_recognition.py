import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load pre-trained Vision Transformer (ViT) for brand recognition
feature_extractor = ViTFeatureExtractor.from_pretrained("models/qwen2/brand_recognition")
model = ViTForImageClassification.from_pretrained("models/qwen2/brand_recognition")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_brand(frame):
    """Predict brand from frame."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = outputs.logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

# Real-time video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    brand = predict_brand(frame)
    cv2.putText(frame, f"Brand: {brand}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Brand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()