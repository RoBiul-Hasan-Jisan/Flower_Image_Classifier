import os
import torch
import requests
from flask import Flask, render_template, request
from torchvision import transforms, models
from PIL import Image

MODEL_URL = "https://drive.google.com/uc?export=download&id=1T722u9sxYjFkAvQFI1xjzJbXTDWbAtxN"
MODEL_PATH = "flower_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
os.makedirs("static", exist_ok=True)  # ✅ ensure static folder

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                downloaded += len(data)
                percent = downloaded / total_size * 100
                print(f"\rDownloading: {percent:.2f}%", end="")
        print("\nModel downloaded!")

def load_model():
    download_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)  # ✅ no deprecation warning
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model, class_names

model, class_names = load_model()

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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file.stream).convert("RGB")
            file_path = os.path.join("static", file.filename)
            image.save(file_path)
            predicted_class, conf = predict(image)
            prediction = predicted_class
            confidence = f"{conf * 100:.2f}%"
            image_url = f"/{file_path}"  # ✅ relative URL

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_url=image_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
