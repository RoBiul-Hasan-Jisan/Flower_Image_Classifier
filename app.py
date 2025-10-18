import os
import torch
import requests
from flask import Flask, render_template, request
from torchvision import transforms, models
from PIL import Image
from io import BytesIO

# -------------------- Config --------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1T722u9sxYjFkAvQFI1xjzJbXTDWbAtxN"
MODEL_PATH = "flower_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# -------------------- Download Model --------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print(" Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                downloaded += len(data)
                percent = downloaded / total_size * 100
                print(f"\rDownloading: {percent:.2f}%", end="")
        print("\n Model downloaded!")

# -------------------- Load Model --------------------
def load_model():
    download_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model, class_names

model, class_names = load_model()

# -------------------- Preprocess & Predict --------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(image: Image.Image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx]
    return predicted_class, confidence.item()

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Read directly from memory (no saving)
            image_bytes = file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            predicted_class, conf = predict(image)
            prediction = predicted_class
            confidence = f"{conf * 100:.2f}%"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_url=None)  # no image displayed

# -------------------- Run --------------------
if __name__ == "__main__":
    # Important for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
