#  Flower Image Classifier (Flask + PyTorch)

This is a simple **Flower Image Classification Web App** built using **Flask**, **PyTorch**, and **Bootstrap** for UI.  
You can upload an image of a flower, and the model will predict its class in real-time — **no images are stored on the server**.



---

##  Features

-  **Deep Learning Model** using ResNet18 trained on a flower dataset  
-  Flask backend for serving predictions  
-  No image storage — processed in memory only  
-  Simple, responsive UI with Bootstrap  
-  Deployable on [Render.com](https://render.com)

---

##  Tech Stack

- **Backend:** Flask (Python)  
- **ML Framework:** PyTorch, Torchvision  
- **Frontend:** HTML, Bootstrap  
- **Deployment:** Render (Gunicorn)

---
##  How It Works

1. **Upload an Image**: Select a flower image (jpg/png) in your browser.  
2. **Backend Processing**: Flask receives the image in memory.  
3. **Preprocessing**: The image is resized to 224x224, converted to a PyTorch tensor, and normalized.  
4. **Prediction**: ResNet18 model predicts the flower class and confidence.  
5. **Result**: The predicted flower name and confidence percentage are displayed on the page.  



