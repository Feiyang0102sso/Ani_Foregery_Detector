import os
import torch
import cv2
import numpy as np
import gradio as gr
from IMDLBenCo.registry import MODELS
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# ================= 1. 加载满级大脑 (模型初始化) =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("⏳ 正在唤醒 AniXplore 满级大脑...")

# 1.1 初始化模型结构 (绕过预训练骨干网络的要求)
model = MODELS.get('AniXplore')(image_size=512, seg_pretrain_path=None)

# 1.2 兼容云端与本地路径
ckpt_path = "checkpoint-8_260306_2018.pth" 
if not os.path.exists(ckpt_path):
    ckpt_path = r"D:\ZWKUS\CPS 4951\AnimeDL2M-main\AniXplore\IMDLBenCo\my_source_model\checkpoint-8.pth"

# 加载权重文件
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# 1.3 剥离 DDP 多卡训练留下的 'module.' 前缀字典映射
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 1.4 加载权重至 GPU 并开启评估模式
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("✅ 模型加载完毕，可以开始检测！")

# ================= 2. 图像预处理流水线 (Data Pipeline) =================
# 严格按照 AniXplore 训练时的图像均值与标准差进行归一化
transform = Compose([
    Resize(512, 512),  
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

SOURCE_MAP = {
    0: "By Human",
    1: "FLUX",
    2: "SDXL",
    3: "SD"
}

# ================= 3. 核心预测与热力图绘制 (Inference & Visualization) =================
def api_predict(image):
    """
    接收前端网页传来的单张 Numpy 图像，返回渲染好的热力图、检测报告、和原始张量
    """
    if image is None:
        return None, "No Image Provided", []
        
    try:
        from PIL import Image
        # The input numpy array from pure frontend via API payload might be in an unpredictable channel order.
        # Explicitly enforce RGB mode discarding potential alpha.
        if image.shape[-1] == 4:
            image = image[..., :3] # RGBA to RGB manually
            
        # VERY IMPORTANT: Gradio's base gr.Image usually provides RGB, 
        # but pure API calls with external file inputs might pass BGR if handled by certain OpenCV wrappers on the frontend.
        # We assume standard RGB payload from the Vue canvas.
        img_pil = Image.fromarray(image.astype('uint8')).convert("RGB")
        image = np.array(img_pil)
        
        # 获取用户上传原图的真实尺寸，方便后续拉伸热力图
        original_h, original_w = image.shape[:2]

        # 洗菜切菜：图像预处理并增加 batch 维度 (H,W,C -> 1,C,H,W)
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

        # The resulting `overlay` is correctly RGB here.
        # But when returning it as a Numpy array/PIL Image via gr.Image API payload, 
        # Gradio API sometimes incorrectly casts the channels depending on origin type.
        # Ensure we return a strict PIL RGB image from the numpy array.
        overlay_pil = Image.fromarray(overlay.astype('uint8'), 'RGB')
        raw_mask_data = raw_mask_resized.tolist()
        
        return overlay_pil, report_data, raw_mask_data
    except Exception as e:
        import traceback
        return None, f"Analysis Error: {str(e)}\n{traceback.format_exc()}", []

# ================= 7. 搭建 Headless API 路由 =================
with gr.Blocks() as app:
    # Use standard components merely as API entry points
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

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7865)
