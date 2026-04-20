import os
import modal
import sys
import traceback

app = modal.App("anixplore-backend")

local_imdl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") 
    .pip_install(
        "torch", 
        "torchvision", 
        "gradio==4.44.0", 
        "huggingface-hub==0.23.2",
        "opencv-python", 
        "albumentations", 
        "numpy", 
        "Pillow",
        "fastapi[standard]",
        "timm",
        "einops"
    )
    .add_local_dir(os.path.join(local_imdl_path, "IMDLBenCo"),
                   remote_path="/workspace/IMDLBenCo")
)

volume = modal.Volume.from_name("anixplore-weights", create_if_missing=True)

sys.path.append("/workspace")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from IMDLBenCo.registry import MODELS
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

SOURCE_MAP = {
    0: "By Human",
    1: "FLUX",
    2: "SDXL",
    3: "SD"
}

_model = None
_transform = None

def get_model():
    global _model, _transform
    if _model is None:
        print("⏳ 正在 Modal GPU 容器内唤醒 AniXplore 满级大脑...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MODELS.get('AniXplore')(image_size=512, seg_pretrain_path=None)
        
        ckpt_path = "/weights/checkpoint-8_260306_2018.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"在 Modal Volume 中没有找到模型权重 {ckpt_path}。请先上传权重！")
            
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        _model = model
        print("✅ Modal 容器内模型加载完毕！")

        _transform = Compose([
            Resize(512, 512),  
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return _model, _transform

def api_predict(image):
    if image is None:
        return None, "No Image Provided", []
        
    try:
        model, transform = get_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if image.shape[-1] == 4:
            image = image[..., :3] 
            
        img_pil = Image.fromarray(image.astype('uint8')).convert("RGB")
        image = np.array(img_pil)
        original_h, original_w = image.shape[:2]

        tensor = transform(image=image)['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            dummy_mask = torch.zeros((1, 1, 512, 512), dtype=torch.float, device=device)
            dummy_label = torch.zeros((1,), dtype=torch.float, device=device)
            preds = model(tensor, dummy_mask, dummy_label)

            mask_logits = None
            source_idx = 0
            source_conf = 0.0
            all_source_probs = None

            if isinstance(preds, dict):
                mask_logits = preds.get('pred_mask', None)
                cls_prob = 0.0
                if 'pred_label_prob' in preds:
                    cls_prob = float(preds['pred_label_prob'].item())
                elif 'pred_label' in preds:
                    cls_prob = float(preds['pred_label'].item())

                if 'pred_source' in preds:
                    source_idx = int(preds['pred_source'].item())

                if 'raw_source_logit' in preds:
                    source_probs = torch.softmax(preds['raw_source_logit'], dim=1)[0]
                    source_conf = float(source_probs[source_idx].item()) * 100
                    all_source_probs = {k: float(source_probs[i].item()) * 100 for i, k in SOURCE_MAP.items()}
                else:
                    source_conf = 99.99

            if mask_logits is None:
                if isinstance(preds, (tuple, list)):
                    for p in preds:
                        if len(p.shape) >= 3:
                            mask_logits = p
                            break
                else:
                    mask_logits = preds

        mask = mask_logits.squeeze().cpu().numpy()

        if mask.ndim == 3:
            mask = mask[0]
        elif mask.ndim == 0:
            mask = np.full((512, 512), mask.item())

        cls_confidence = cls_prob * 100
        mean_score = float(np.mean(mask)) * 100
        peak_score = float(np.percentile(mask, 99.5)) * 100
        suspicious_area_ratio = np.sum(mask > 0.5) / mask.size

        if suspicious_area_ratio > 0.01:
            mask_confidence = peak_score
        else:
            mask_confidence = mean_score

        final_confidence = max(cls_confidence, mask_confidence)
        is_whole_image_ai = (cls_confidence > 50) and (suspicious_area_ratio < 0.01)

        if is_whole_image_ai:
            uniform_mask = np.full((original_h, original_w), cls_prob)
            mask_resized = uniform_mask
            raw_mask_resized = uniform_mask

            heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            alpha = np.full((original_h, original_w, 1), 0.4 * cls_prob)
            overlay = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        else:
            clean_mask = np.where(mask > 0.5, mask, 0)
            mask_resized = cv2.resize(clean_mask, (original_w, original_h))
            raw_mask_resized = cv2.resize(mask, (original_w, original_h))

            heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            alpha = np.where(mask_resized > 0, 0.4, 0)[..., np.newaxis]
            overlay = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)

        source_name = SOURCE_MAP.get(source_idx, "未知来源")

        if final_confidence >= 70:
            danger_level = 'danger'
        elif 45 <= final_confidence < 70:
            danger_level = 'warning'
        else:
            danger_level = 'safe'

        report_data = {
            "overall_ai_prob": cls_confidence,
            "local_tamper_prob": mask_confidence,
            "final_decision_score": final_confidence,
            "danger_level": danger_level,
            "is_whole_image_ai": bool(is_whole_image_ai),
            "source_name": source_name,
            "source_conf": source_conf,
            "all_source_probs": all_source_probs if all_source_probs else {}
        }

        overlay_pil = Image.fromarray(overlay.astype('uint8'), 'RGB')
        raw_mask_data = raw_mask_resized.tolist()
        
        return overlay_pil, report_data, raw_mask_data
    except Exception as e:
        return None, f"Analysis Error: {str(e)}\n{traceback.format_exc()}", []

def create_gradio_app():
    with gr.Blocks() as interface:
        img_input = gr.Image(type="numpy")
        img_overlay = gr.Image(type="pil")
        txt_report = gr.JSON()
        json_mask = gr.JSON()
        
        btn = gr.Button("API Trigger", visible=False)
        btn.click(
            fn=api_predict,
            inputs=[img_input],
            outputs=[img_overlay, txt_report, json_mask],
            api_name="predict"
        )
    return interface

@app.function(
    image=image,
    gpu="T4",
    memory=8192, 
    volumes={"/weights": volume}, 
)
@modal.asgi_app()
def fastapi_app():
    import fastapi
    from fastapi.middleware.cors import CORSMiddleware
    
    print("🚀 [STARTUP LOG] 正在初始化 FastAPI 服务器...")
    try:
        web_app = fastapi.FastAPI()
        
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        print("🚀 [STARTUP LOG] 正在创建 Gradio 原生界面...")
        interface = create_gradio_app()
        
        print("🚀 [STARTUP LOG] 正在将 Gradio 挂载为 ASGI 子路由...")
        return gr.mount_gradio_app(web_app, interface, path="/")
    except Exception as e:
        print("❌ [CRITICAL ERROR] 容器启动遭遇致命错误:")
        traceback.print_exc(file=sys.stdout)
        raise e
