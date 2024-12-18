import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load fine-tuned Qwen-2 model
model = torch.load("models/qwen2/freshness_detection/model_fresh.pt", map_location=torch.device('cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_freshness(frame):
    """Predict freshness (fresh/rotten) from frame."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    return "Fresh" if predicted.item() == 0 else "Rotten"

# Real-time video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = predict_freshness(frame)
    cv2.putText(frame, f"Freshness: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Freshness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()