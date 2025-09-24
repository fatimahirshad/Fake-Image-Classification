# ==========================
# app.py (Gradio UI) - Gradio 4.29 compatible
# ==========================

import gradio as gr
from model_inference import model, transform, CLASS_NAMES, DEVICE
import torch
import torch.nn.functional as F
from PIL import Image
import time

def classify(image):
    # Optional delay so scanning animation is visible
    time.sleep(2)  

    # Preprocess
    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    # Make dictionary {class: prob}
    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return results, gr.update(visible=False)  # hide scanning box after prediction


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
            submit_btn = gr.Button("🕵️ Detect", variant="primary")
        with gr.Column(scale=1):
            # Neon blue glowing prediction box
            gr.HTML(
                """
                <div style="
                    border: 2px solid;
                    border-image: linear-gradient(45deg, #00f, #00eaff) 1;
                    border-radius: 15px;
                    padding: 12px;
                    background-color: #0a0a0a;
                    color: #f5f5f5;
                    box-shadow: 0 0 20px #00eaff;
                ">
                    <h3>📊 Prediction (Top 3)</h3>
                </div>
                """
            )
            output_label = gr.Label(num_top_classes=3, label="Prediction")

            # 🔵 Scanning animation box (hidden by default)
            scan_box = gr.HTML(
                """
                <div id="scanbox" style="position: relative; width: 100%; height: 180px; 
                    background: linear-gradient(to bottom, #002244, #0044aa); 
                    border-radius: 12px; 
                    overflow: hidden; 
                    border: 2px solid #00eaff; 
                    box-shadow: 0 0 20px #00eaff;">
                    
                    <div style="position: absolute; top: -100%; left: 0; right: 0; bottom: 0;
                        background: linear-gradient(to bottom, rgba(0,255,255,0.8), rgba(0,255,255,0));
                        animation: scanmove 2s linear infinite;"></div>
                    
                    <h3 style="position: absolute; top: 50%; left: 50%; 
                        transform: translate(-50%, -50%);
                        color: #00ffcc; font-family: monospace; text-shadow: 0 0 10px #00eaff;">
                        🔍 Scanning...
                    </h3>
                </div>

                <style>
                @keyframes scanmove {
                    from { top: -100%; }
                    to { top: 100%; }
                }
                </style>
                """,
                visible=False,
            )

    # Add example images
    gr.Examples(
        examples=examples,
        inputs=input_img,
        label="Try with example images"
    )

    # 📦 About Model Section (neon green styled box)
    gr.HTML(
        """
        <div style="
            border: 2px solid #39FF14; 
            border-radius: 15px; 
            padding: 15px; 
            background-color: #0d0d0d; 
            color: #f5f5f5;
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

    # 🔗 Button workflow: show scan box → run classify → hide scan box + show results
    submit_btn.click(
        fn=lambda x: (gr.update(visible=True), x),
        inputs=input_img,
        outputs=[scan_box, input_img],
    ).then(
        fn=classify,
        inputs=input_img,
        outputs=[output_label, scan_box],
    )

# --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    demo.launch()
