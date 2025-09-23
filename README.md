---
title: Deepfake Detector 🕵️
emoji: 👀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🕵️ Deepfake Detector (not a pretrained model)

Detect whether an uploaded face image is **real** or a type of **deepfake** using a trained CNN model.  

## 📌 Features
- Upload any face image (`.jpg`, `.png`, `.jpeg`)  
- Get prediction results with confidence scores  
- Supports multiple deepfake categories (based on FaceForensics++ dataset)  
- Simple web app built with **Gradio** and deployed on **Hugging Face Spaces**

---

## 🚀 How to Use
1. Click **Upload Image** and choose a face image.  
2. Press **🔍 Analyze**.  
3. The app will return prediction probabilities for:  
   - **Original**  
   - **Deepfake categories** (Deepfakes, Face2Face, FaceSwap, NeuralTextures, etc.)

---

## 🛠️ Project Structure
deepfake-detector/
  - app.py                (Gradio UI) 
  - model_inference.py    (Model architecture + predict function)
  - best_model.pth        (Trained model weights)
  - requirements.txt      (Dependencies)
  - README.md           

---

## 📦 Installation (Run Locally)
```bash
# Clone repo
git clone https://huggingface.co/spaces/YOUR-USERNAME/deepfake-detector
cd deepfake-detector

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py