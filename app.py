# ==========================
# app.py (Gradio UI with scanning overlay + neon boxes)
# ==========================

import gradio as gr
from model_inference import model, transform, CLASS_NAMES, DEVICE
import torch
import torch.nn.functional as F
from PIL import Image
import time

# --------------------------
# Inference function
# --------------------------
def classify(image):
    # Simulate scanning time (overlay visible during this)
    time.sleep(2)

    # Preprocess and run model
    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return results, ""  # clear overlay after prediction


# --------------------------
# Build UI
# --------------------------
examples = [
    ["example_images/deepfakesf0.jpg"],
    ["example_images/original.jpg"],
    ["example_images/face2face_f4.jpg"],
    ["example_images/faceshfifter_f4.jpg"],
]

with gr.Blocks(theme=gr.themes.Soft(), css="""
/* Scanning overlay effect */
.scan-box {
    position: relative;
    display: inline-block;
    width: 100%;
}
.scan-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        to bottom,
        rgba(0, 255, 255, 0.25),
        rgba(0, 255, 255, 0.25) 2px,
        transparent 2px,
        transparent 4px
    );
    animation: scanmove 2s linear infinite;
    pointer-events: none;
    border-radius: 10px;
}
@keyframes scanmove {
    0% { background-position: 0 -100%; }
    100% { background-position: 0 100%; }
}
""") as demo:
    gr.Markdown(
        """
        # 🕵️ Deepfake Detector  
        Upload a face image and the model will tell you if it's **Original** or a type of **Deepfake**.  
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row(elem_classes="scan-box"):
                input_img = gr.Image(type="numpy", label="Upload Image", sources=["upload"])
                overlay_html = gr.HTML("", elem_id="overlay")
            submit_btn = gr.Button("🔍 Detect", variant="primary")

        with gr.Column(scale=1):
            # Neon blue glowing prediction box
            gr.HTML(
                """
                <div style="
                    border: 2px solid;
                    border-image: linear-gradient(45deg, #00f, #00eaff) 1;
                    border-radius: 15px;
                    padding: 12px;
                    background: transparent;
                    color: inherit;
                    box-shadow: 0 0 20px #00eaff;
                ">
                    <h3>📊 Prediction (Top 3)</h3>
                </div>
                """
            )
            output_label = gr.Label(num_top_classes=3, label="Prediction")

    # Example images
    gr.Examples(
        examples=examples,
        inputs=input_img,
        label="Try with example images"
    )

    # 📦 About Model Section (neon green box)
    gr.HTML(
        """
        <div style="
            border: 2px solid #39FF14; 
            border-radius: 15px; 
            padding: 15px; 
            background: transparent; 
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

    # --------------------------
    # Event binding
    # --------------------------
    def wrap_with_overlay(img):
        # Show overlay immediately
        return classify(img)[0], """<div class='scan-overlay'></div>"""

    submit_btn.click(
        fn=wrap_with_overlay,
        inputs=input_img,
        outputs=[output_label, overlay_html]
    )

# --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    demo.launch()
