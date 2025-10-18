import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
import requests


MODEL_URL = "https://drive.google.com/uc?export=download&id=1T722u9sxYjFkAvQFI1xjzJbXTDWbAtxN"
MODEL_PATH = "flower_classifier.pth"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        st.info("Downloading model, please wait...")
        response = requests.get(MODEL_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(MODEL_PATH, "wb") as f:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                downloaded += len(data)
                st.progress(min(downloaded / total_size, 1.0))
        st.success("Model downloaded!")

@st.cache_resource(show_spinner=False)
def load_model():
    download_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    return model, class_names

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def predict(image: Image.Image, model, class_names):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx]
    return predicted_class, confidence.item()

def main():
    st.title("🌸 Flower Image Classifier")
    st.write("Upload a flower image and the model will predict its class.")

    model, class_names = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            predicted_class, confidence = predict(image, model, class_names)
            st.success(f"Prediction: **{predicted_class}** (Confidence: {confidence * 100:.2f}%)")

if __name__ == "__main__":
    main()
