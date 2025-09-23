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
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🕵️ Deepfake Detector  
        Upload a face image and the model will tell you if it's **Original** or a type of **Deepfake**.  
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Upload Image")
            submit_btn = gr.Button("🔍 Analyze", variant="primary")
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=3, label="Prediction (Top 3)")

    # Link button to function
    submit_btn.click(fn=classify, inputs=input_img, outputs=output_label)

# --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    demo.launch()
