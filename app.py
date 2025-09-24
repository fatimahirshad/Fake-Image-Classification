# ==========================
# app.py (Gradio UI)
# ==========================

import gradio as gr
from model_inference import model, transform, CLASS_NAMES, DEVICE
import torch
import torch.nn.functional as F
from PIL import Image

def classify(image):
    # Preprocess
    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    # Make dictionary {class: prob}
    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return results

# --------------------------
# Build UI
# --------------------------
examples = [
    ["example_images/deepfakesf0.jpg"],
    ["example_images/original.jpg"],
    ["example_images/face2face_f4.jpg"],
    ["example_images/faceshfifter_f4.jpg"],
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🕵️ Deepfake Detector  
        Upload a face image and the model will tell you if it's **Original** or a type of **Deepfake**.  
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Upload Image", sources=["upload"])
            submit_btn = gr.Button("🔍 Detect", variant="primary")
        with gr.Column(scale=1):
            # Neon blue glowing prediction box (transparent background)
            gr.HTML(
                """
                <div style="
                    border: 2px solid;
                    border-image: linear-gradient(45deg, #00f, #00eaff) 1;
                    border-radius: 15px;
                    padding: 12px;
                    background-color: transparent;
                    color: inherit;
                    box-shadow: 0 0 20px #00eaff;
                ">
                    <h3>📊 Prediction (Top 3)</h3>
                </div>
                """
            )
            output_label = gr.Label(num_top_classes=3, label="Prediction")

    # Add example images
    gr.Examples(
        examples=examples,
        inputs=input_img,
        label="Try with example images"
    )

    # 📦 About Model Section (neon green styled, transparent background)
    gr.HTML(
        """
        <div style="
            border: 2px solid #39FF14; 
            border-radius: 15px; 
            padding: 15px; 
            background-color: transparent; 
            color: inherit;
            box-shadow: 0 0 15px #39FF14;
        ">
            <h3>📦 About the Model</h3>
            <p>🔹 Trained on <b>FaceForensics++ (FF++) dataset</b></p>
            <p>🔹 Detects <b>Deepfakes, FaceSwap, Face2Face, Neural Textures and Original</b></p>
            <p>🔹 Achieved <b>92% accuracy</b> on unseen test data</p>

            <h4>💡 Tips for Best Results</h4>
            <ul>
                <li>Upload clear frontal face images</li>
                <li>Avoid low-light / blurred images</li>
                <li>Works best on single-face photos</li>
            </ul>
        </div>
        """
    )

    # Link button to function
    submit_btn.click(fn=classify, inputs=input_img, outputs=output_label)

# --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    demo.launch()
