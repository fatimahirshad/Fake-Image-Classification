#  Fake Image Classification

A deep learning project to detect **fake (deepfake/manipulated)** vs **real (original)** images using a **Convolutional Neural Network (CNN)** trained on the [**FaceForensics++ C32**](https://github.com/ondyari/FaceForensics)

Includes a **Gradio web app** with a neon-themed UI and a **scanning effect** that simulates image analysis.

***

##  Features

* **Detects** **Deepfake**, **FaceSwap**, **Face2Face**, **Neural Textures**, and **Original** images.
* **Interactive Gradio app** with a vibrant neon design.
* **Prediction box** showing the top 3 results and their confidence scores.
* **Scanning animation** displayed while the image is being analyzed.
* **Example images** included in the repo for quick testing.

***

##  Dataset

This model was trained on the large-scale [**FaceForensics++ C32**](https://github.com/ondyari/FaceForensics).
And from this dataset I have made 2 datasets

> **Note:** Due to size, the full dataset is **not included** in this repository.

👉 1. [**FaceForensics++ Extracted Dataset (C23)**](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23) In this dataset, i extracted frames 5/video from 6 classes (Deepfakes, Face2Face,FaceShifter, FaceSwap, NeuralTextures, Original), **TOTAL= 30k frames**.

👉 2. [**FaceForensics++ C32 Frames (Cropped+Aligned)**](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-c32-frames-cropped-aligned) In this dataset. i just cropped and alingned these frames from above dataset and use in building this model. 


***

##  Demo

Upload an image or test with included examples (`<img width="778" height="571" alt="image" src="https://github.com/user-attachments/assets/b445b75d-b58c-4687-a6cc-00d81f87fddd" />
`).

The app will show a **scanning effect** and then return prediction results.


***

##  Deployment on Hugging Face Spaces

The project is live on **Hugging Face Spaces** 🚀, allowing you to test the model directly in your browser without any installation.

👉 [**Try Fake Image Classification App**](https://huggingface.co/spaces/fatima-irshad/deepfake-detector)


***

##  Credits

* **FaceForensics++** dataset
* Built with **PyTorch** + **Gradio**
* Author: **Fatima Irshad**

***

##  Project Structure
deepfake-detector/
  - app.py                (Gradio UI) 
  - model_inference.py    (Model architecture + predict function)
  - best_model.pth        (Trained model weights)
  - requirements.txt      (Dependencies)
  - README.md

***

##  Installation (Run Locally)
```bash
# Clone repo
git clone https://huggingface.co/spaces/YOUR-USERNAME/deepfake-detector
cd deepfake-detector

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
