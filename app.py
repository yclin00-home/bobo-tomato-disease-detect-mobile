import json
import os

import cv2
import gradio as gr
import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms

MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
MAX_SIDE = 1024      # 圖片最長邊超過這個值就縮小
MIN_CONF = 0.60      # 信心度低於此值 -> 顯示「不確定」

# ---------- 載入模型 ----------
with open("class_names_mnv4.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

device = torch.device("cpu")
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load("tomato_mnv4.pth", map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------- 病斑面積比例 (OpenCV) ----------
def calc_infection_ratio(img_rgb):
    img_hsv = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(img_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    mask_diseased = cv2.inRange(img_hsv, np.array([10, 30, 30]), np.array([35, 255, 255]))
    leaf_mask = cv2.bitwise_or(mask_green, mask_diseased)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    mask_dark = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    lesion = cv2.bitwise_or(mask_diseased, mask_dark)
    lesion = cv2.bitwise_and(lesion, lesion, mask=leaf_mask)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)

    leaf_px = cv2.countNonZero(leaf_mask)
    if leaf_px == 0:
        return 0.0
    return cv2.countNonZero(lesion) / leaf_px * 100.0


# ---------- 診斷 ----------
def diagnose(image_path):
    if image_path is None:
        return None, "N/A", "📷 Please take or upload a photo."

    try:
        img = Image.open(image_path)
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))     # JPEG 解碼時就縮小，省記憶體
        img = ImageOps.exif_transpose(img)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        img = img.convert("RGB")
    except Exception:
        return None, "N/A", "❌ Cannot read this image. Please try another one."

    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)

    conf = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    top = max(conf, key=conf.get)
    top_conf = conf[top]
    name = top.lower()

    if top_conf < MIN_CONF:
        return conf, "N/A", (f"❓ Not sure (best guess: {top}, {top_conf*100:.1f}%). "
                             "Try a clearer, closer photo of a single leaf.")

    if name.startswith("non"):
        return conf, "N/A", "🚫 **This does not look like a tomato leaf.** Please photograph a tomato leaf."

    if "healthy" in name:
        return conf, "N/A", f"✅ The leaf looks **healthy** (confidence {top_conf*100:.1f}%)."

    if "late" in name and "blight" in name:
        ratio = calc_infection_ratio(np.array(img))
        if ratio < 15:
            msg = (f"💡 Mild late blight (lesions cover {ratio:.1f}%). "
                   "Remove infected leaves promptly and apply an organic copper fungicide.")
        elif ratio < 45:
            msg = (f"⚠️ Moderate late blight (lesions cover {ratio:.1f}%). "
                   "Isolate the plant and use a systemic fungicide to limit spread.")
        else:
            msg = (f"🚨 Severe late blight (lesions cover {ratio:.1f}%)! "
                   "Remove and destroy the plant to prevent spore spread.")
        return conf, f"{ratio:.1f}%", msg

    return conf, "N/A", (f"⚠️ Another tomato disease detected (confidence {top_conf*100:.1f}%). "
                         "Please check the symptoms and follow the recommended control measures.")


# ---------- 網頁介面 ----------
with gr.Blocks(title="Tomato Leaf Disease Diagnosis") as demo:
    gr.Markdown("# 🍅 Tomato Leaf Disease Diagnosis")
    gr.Markdown("Take a photo of a tomato leaf, or upload one.")

    image_input = gr.Image(type="filepath", label="Take or upload a photo",
                           sources=["upload", "webcam"])
    btn = gr.Button("🔍 Diagnose", variant="primary")

    label_output = gr.Label(label="Prediction & confidence")
    ratio_output = gr.Textbox(label="Lesion area ratio (late blight only)")
    advice_output = gr.Markdown()

    btn.click(diagnose, inputs=[image_input],
              outputs=[label_output, ratio_output, advice_output])

#demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
demo.launch(share=True)
