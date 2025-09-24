import gradio as gr
from model_inference import model, transform, CLASS_NAMES, DEVICE
import torch
import torch.nn.functional as F
from PIL import Image
import time

def classify(image):
    # Show scanning during this time
    time.sleep(2)

    image = Image.fromarray(image).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return results, ""  # return prediction + empty overlay (hide)

with gr.Blocks(theme=gr.themes.Soft(), css="""
/* Scanning overlay effect */
.scan-box {
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
    border-radius: 10px;
}
@keyframes scanmove {
    0% { background-position: 0 -100%; }
    100% { background-position: 0 100%; }
}
""") as demo:
    gr.Markdown("# 🕵️ Deepfake Detector")

    with gr.Row():
        with gr.Column():
            with gr.Row(elem_classes="scan-box"):
                input_img = gr.Image(type="numpy", label="Upload Image")
                overlay_html = gr.HTML("", elem_id="overlay")
            submit_btn = gr.Button("🔍 Detect", variant="primary")

        with gr.Column():
            gr.HTML(
                """
                <div style="border: 2px solid;
                            border-image: linear-gradient(45deg, #00f, #00eaff) 1;
                            border-radius: 15px;
                            padding: 12px;
                            background: transparent;
                            color: inherit;
                            box-shadow: 0 0 20px #00eaff;">
                    <h3>📊 Prediction (Top 3)</h3>
                </div>
                """
            )
            output_label = gr.Label(num_top_classes=3, label="Prediction")

    # Trigger function: scanning overlay appears while model runs
    def wrap_with_overlay(img):
        return classify(img)[0], """<div class='scan-overlay'></div>"""

    submit_btn.click(
        fn=wrap_with_overlay,
        inputs=input_img,
        outputs=[output_label, overlay_html]
    )

# Run
if __name__ == "__main__":
    demo.launch()
