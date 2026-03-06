import os
os.environ['OMP_NUM_THREADS']=str(1)
from typing import Dict, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import IMDLBenCo.model_zoo as model_zoo


class ImageResizeCustom:
    def __init__(self,
                 interpolation=cv2.INTER_LINEAR):
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.shape[:2]
        length = 512
        if h > w:
            new_h = length
            new_w = int(w * length / h)
        else:
            new_w = length
            new_h = int(h * length / w)
        img = cv2.resize(img, (new_w, new_h),
                         interpolation=self.interpolation)
        # padding
        top = (length - new_h) // 2
        bottom = length - new_h - top
        left = (length - new_w) // 2
        right = length - new_w - left
        img = cv2.copyMakeBorder(img,
                                 top,
                                 bottom,
                                 left,
                                 right,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
        self.padding_512 = (top, bottom, left, right)
        return img

class BaseTransform_Val:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        
    def __call__(self, img) -> torch.Tensor:
        return self.transform(img)

def get_model(device: str = "cpu"):
    num_classes = 15
    model_cfg = dict(
        type='AniXplore',
        in_channels=3,
        num_classes=num_classes,
        encoder_name='convnext_tiny',
        mvt_name='mit_b1',
        device=device
    )
    import collections
    
    model = model_zoo.build_source_model(model_cfg)
    
    # Load weights via native torch load
    ckpt_path = "checkpoint-8_260306_2018.pth" # Needs to be uploaded to HF
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract state_dict (handles nested 'model' keys commonly saved in training)
    state_dict = checkpoint.get("model", checkpoint)
    
    # Remove 'module.' prefixes which are an artifact of DistributedDataParallel (DDP)
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    model.to(device)
    return model

def load_data(img_np: np.ndarray):
    resize_trans = ImageResizeCustom()
    img_512 = resize_trans(img_np)
    padding_512 = resize_trans.padding_512
    # convert to rgb
    img_512 = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB)
    
    val_trans = BaseTransform_Val()
    img_512_tensor = val_trans(img_512)
    return img_512_tensor, padding_512

def convert_label(label: int) -> str:
    source_map = {
        0: "真实人类手绘 (Real)",
        1: "FLUX 生成",
        2: "SDXL 生成",
        3: "Stable Diffusion (SD) 生成"
    }
    return source_map.get(label, 'Unknown AI Engine')


print("Initializing Headless Backend...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = get_model(DEVICE)
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Failed to load model logic. Error: {e}")
    # We still define logic so Gradio won't crash instantly, but it will fail on predict
    model = None

@torch.no_grad()
def headless_predict(img):
    if img is None:
        return None, "No image provided.", []
        
    try:
        img_np = np.array(img)
        img_512_tensor, padding_512 = load_data(img_np)
        img_512_tensor = img_512_tensor.unsqueeze(0).to(DEVICE)
        
        # Inference
        predicts = model(img_512_tensor)
        
        # Process results
        cls_logits = predicts['pred_label']
        mask_logits = predicts['pred_mask'].cpu()
        
        pred_cls = torch.argmax(cls_logits, dim=1).cpu().numpy()[0]
        pred_prob = torch.softmax(cls_logits, dim=1).cpu().numpy()[0][pred_cls]
        
        pred_mask = torch.sigmoid(mask_logits).numpy()[0, 0]
        
        # Remove padding from mask
        top, bottom, left, right = padding_512
        mask_no_pad = pred_mask[top:512-bottom, left:512-right]
        
        # Resize mask back to original image size
        orig_h, orig_w = img_np.shape[:2]
        pred_mask_full = cv2.resize(mask_no_pad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Generate Heatmap (JET)
        heatmap = cv2.applyColorMap(np.uint8(255 * (1.0 - pred_mask_full)), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend Heatmap with original
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        overlay_pil = Image.fromarray(overlay)
        
        # Generate Text Report
        p_str = f"{(pred_prob*100):.2f}%"
        report_lines = []
        is_forge = (pred_prob > 0.5)

        if not is_forge:
             report_lines.append("✅ SAFE: Highly likely authentic human hand-drawn art.\n")
             report_lines.append(f"▸ AI Gen Probability: {p_str}")
             report_lines.append("▸ Detected Source Model: Authentic Art")
        else:
            if pred_mask_full.mean() > 0.7:
                 report_lines.append("🚨 DANGER: Highly suspected WHOLE Image AI Generation!\n")
            else:
                 report_lines.append("🚨 DANGER: Highly suspected PARTIAL AI Inpainting!\n")
            report_lines.append(f"▸ AI Gen Probability: {p_str}")
            report_lines.append(f"▸ Detected Source Model: {convert_label(pred_cls)}")

        report_str = "\n".join(report_lines)
        
        # Pass pure probabilities for the frontend probe
        raw_mask_data = pred_mask_full.tolist() 

        return overlay_pil, report_str, raw_mask_data
        
    except Exception as e:
         return None, f"Analysis Error: {str(e)}", []


# Define Headless Gradio API without complex UI blocks
with gr.Blocks() as app:
    # Use standard components merely as API entry points
    img_input = gr.Image(type="pil")
    img_overlay = gr.Image()
    txt_report = gr.Textbox()
    json_mask = gr.JSON()
    
    # We do NOT add UI elements or buttons to build an exposed view.
    # The Gradio Blocks router simply exposes a `/predict` API route automatically.
    
    # We create a hidden API binding
    btn = gr.Button("API Trigger", visible=False)
    btn.click(
        fn=headless_predict,
        inputs=[img_input],
        outputs=[img_overlay, txt_report, json_mask],
        api_name="predict" # MUST be /predict
    )

if __name__ == '__main__':
    # Launch for HuggingFace (CORS enabled)
    app.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
