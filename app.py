# ==========================
# app.py (Gradio UI with auto-hide scanning effect)
# ==========================

import gradio as gr
from model_inference import model, transform, CLASS_NAMES, DEVICE
import torch
import torch.nn.functional as F
from PIL import Image
import time

def classify(image):
    # Simulate processing delay for scanning effect
    time.sleep(2)

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

with gr.Blocks(theme=gr.themes.Soft(), css="""
/* Scanning overlay effect */
.scan-container {
    position: relative;
    display: inline-block;
}
.scan-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        to bottom,
        rgba(0, 255, 255, 0.2),
        rgba(0, 255, 255, 0.2) 2px,
        transparent 2px,
        transparent 4px
    );
    animation: scanmove 2s linear infinite;
    pointer-events: none;
    display: none;
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
            # Upload image + scanning overlay
            with gr.Row():
                input_img = gr.Image(type="numpy", label="Upload Image", elem_id="scan-image", sources=["upload"])
                scan_overlay = gr.HTML(
                    """<div class="scan-overlay" id="scan-overlay"></div>""",
                    elem_id="scan-box"
                )
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
                    background-color: transparent;
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

    # 📦 About Model Section (green neon box)
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

    # JS: show scanning overlay on click, hide after prediction
    submit_btn.click(
        fn=classify,
        inputs=input_img,
        outputs=output_label,
        _js="""
        (img) => {
            const overlay = document.getElementById("scan-overlay");
            if (overlay) overlay.style.display = "block"; // show on detect
            return img;
        }
        """,
        postprocess=False
    )

    # Hide overlay after results show
    output_label.change(
        lambda x: x,
        inputs=output_label,
        outputs=output_label,
        _js="""
        () => {
            const overlay = document.getElementById("scan-overlay");
            if (overlay) overlay.style.display = "none"; // hide after prediction
        }
        """
    )

# --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    demo.launch()
