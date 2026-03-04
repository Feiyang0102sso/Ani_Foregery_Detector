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

# 1.2 加载权重文件 (设置 weights_only=False 绕过 PyTorch 2.6 的安全拦截)
ckpt = torch.load("CKPTS/AniXplore/checkpoint-29.pth", map_location=device, weights_only=False)

# 1.3 剥离 DDP 多卡训练留下的 'module.' 前缀字典映射
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 🚨 新增拦截逻辑：强行删除形状不匹配的废弃权重 (解决 auto_weight.params 报错)
keys_to_delete = [k for k in state_dict.keys() if 'auto_weight' in k]
for k in keys_to_delete:
    del state_dict[k]
if keys_to_delete:
    print(f"🧹 已清理训练专用的无用参数: {keys_to_delete}")

# 1.4 加载权重至 GPU 并开启评估模式 (切断梯度计算，节省显存)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("✅ 模型加载完毕，可以开始检测！")

# ================= 2. 图像预处理流水线 (Data Pipeline) =================
# 严格按照 AniXplore 训练时的图像均值与标准差进行归一化
transform = Compose([
    Resize(512, 512),  # 强制缩放到 512x512，与模型输入层对齐
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ================= 3. 核心预测与热力图绘制 (Inference & Visualization) =================
def predict_image(image):
    """
    接收前端网页传来的单张 Numpy 图像，返回渲染好的热力图和检测报告
    """
    # 获取用户上传原图的真实尺寸，方便后续拉伸热力图
    original_h, original_w = image.shape[:2]

    # 洗菜切菜：图像预处理并增加 batch 维度 (H,W,C -> 1,C,H,W)
    tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    with torch.no_grad():  # 禁用梯度引擎，极大提升推理速度
        # 3.1 伪造占位符标签：骗过原学术代码中对 forward(mask, label) 的强制参数检查
        dummy_mask = torch.zeros((1, 1, 512, 512), dtype=torch.float, device=device)
        dummy_label = torch.zeros((1,), dtype=torch.float, device=device)

        # 3.2 运行前向传播 (Forward Pass)
        preds = model(tensor, dummy_mask, dummy_label)

        # 3.3 特征图自适应提取：智能寻找模型返回的 2D 像素级预测矩阵
        mask_logits = None

        if isinstance(preds, dict):
            # 获取篡改热力图预测
            mask_logits = preds.get('pred_mask', None)

        # 兼容旧版本的列表/元组返回格式
        if mask_logits is None:
            if isinstance(preds, (tuple, list)):
                for p in preds:
                    if len(p.shape) >= 3:  # 寻找带有 [Batch, Channel, H, W] 的张量
                        mask_logits = p
                        break
            else:
                mask_logits = preds

    # 3.4 提取真实概率图：撤销错误的双重 Sigmoid，直接接收模型的合法概率输出！
    mask = mask_logits.squeeze().cpu().numpy()

    # 防御性维度对齐：确保输出是一个纯粹的 512x512 二维矩阵
    if mask.ndim == 3:
        mask = mask[0]
    elif mask.ndim == 0:
        mask = np.full((512, 512), mask.item())

    # === 4. 统计学噪声抑制与鲁棒性打分 (Statistical Noise Mitigation) ===
    # 计算全局平均可疑度（正常真图的底噪会被稀释到极低）
    mean_score = float(np.mean(mask)) * 100

    # 计算局部最高峰值（使用 99.5% 分位数，过滤掉极个别神经质跳跃的噪点像素）
    peak_score = float(np.percentile(mask, 99.5)) * 100

    # 连通域面积计算：严格以 0.5 (50%) 为分水岭，统计疑似假图像素的占比
    suspicious_area_ratio = np.sum(mask > 0.5) / mask.size

    # 双轨制打分策略 (Dual-track Scoring Strategy)
    if suspicious_area_ratio > 0.01:
        final_confidence = peak_score
    else:
        final_confidence = mean_score  # 若面积太小纯属底噪，则打回原形采用平均分

    # === 5. 视觉可视化 (Visualization Rendering) ===
    # 视觉净化：只要概率低于 0.5 (即模型认为倾向于真)，全部抹零归为安全，画面干干净净
    clean_mask = np.where(mask > 0.5, mask, 0)

    # 将 512x512 的特征图拉伸回用户原图的真实比例
    mask_resized = cv2.resize(clean_mask, (original_w, original_h))

    # 将原始概率图也拉伸，用于精确查询（不经过阈值过滤，保留所有细节）
    raw_mask_resized = cv2.resize(mask, (original_w, original_h))

    # 应用伪彩映射 (COLORMAP_JET: 蓝 -> 绿 -> 红)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 智能叠图蒙版：只有在预测矩阵大于 0 的地方，才以 40% 的透明度覆盖热力图
    alpha = np.where(mask_resized > 0, 0.4, 0)[..., np.newaxis]
    overlay = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    # === 6. 三级决策报告生成 (Tri-level Warning System) ===
    if final_confidence >= 70:
        report = f"🚨 危险：高度疑似 AI 生成或局部重绘！\n"
    elif 45 <= final_confidence < 70:
        report = f"⚠️ 可疑：局部存在未知修改痕迹。\n"
    else:
        report = f"✅ 安全：大概率为真实人类手绘作品。\n"

    report += f"▸ 篡改综合置信度: {final_confidence:.2f}%"

    # 返回叠加图、报告、以及原始概率图供交互查询
    return overlay, report, raw_mask_resized


# ================= 7. 搭建 Web UI 界面 (Gradio App) =================
def get_pixel_confidence(evt: gr.SelectData, mask_data):
    """
    响应图片点击事件，返回点击位置的置信度
    """
    if mask_data is None:
        return "⚠️ 请先上传图片并点击检测，生成热力图后再试。"

    # evt.index 返回的是 (x, y) 坐标
    x, y = evt.index

    # 确保坐标在数组范围内
    h, w = mask_data.shape
    if 0 <= y < h and 0 <= x < w:
        confidence = mask_data[y, x] * 100
        return f"📍 坐标 ({x}, {y}) 置信度: {confidence:.2f}%"
    else:
        return f"❌ 坐标 ({x}, {y}) 超出范围 ({w}x{h})"


def clear_all():
    """
    清除所有输入输出内容
    """
    return None, None, "", "👆 点击上方热力图任意位置，查看该点的 AI 篡改置信度...", None


# 构建现代化交互面板 (使用 Blocks 以支持更灵活的布局和事件)
with gr.Blocks(title="AniXplore 图像篡改鉴证系统 (v1.0)", theme="default") as interface:
    gr.Markdown("# 🕵️ AniXplore 图像篡改鉴证系统 (v1.0)")
    gr.Markdown(
        "基于 Shearlet 剪切波变换的深层伪造定位技术。上传一张图片，系统将秒级揪出 AI 局部重绘 (Inpainting) 痕迹。")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📸 图像输入区 (上传你的动漫原图或假图)", type="numpy")
            with gr.Row():
                predict_btn = gr.Button("🚀 开始检测", variant="primary", scale=3)
                clear_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)

        with gr.Column(scale=1):
            # 输出图片组件，允许点击选择
            output_image = gr.Image(label="🔥 频域异常高亮区 (点击热力图可查看具体置信度)")

            # 显示点击位置置信度的文本框
            pixel_info = gr.Textbox(label="🔍 像素级探针", value="👆 点击上方热力图任意位置，查看该点的 AI 篡改置信度...",
                                    interactive=False)

            # 报告输出框
            report_box = gr.Textbox(label="📊 AI 智能鉴证报告", lines=3)

    # 隐藏的 State 组件，用于存储原始概率图
    mask_state = gr.State()

    # 按钮点击事件：触发预测，更新图片、报告和 State
    predict_btn.click(
        fn=predict_image,
        inputs=[input_image],
        outputs=[output_image, report_box, mask_state]
    )

    # 清除按钮点击事件
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[input_image, output_image, report_box, pixel_info, mask_state]
    )

    # 图片点击事件：触发探针函数，更新置信度显示
    output_image.select(
        fn=get_pixel_confidence,
        inputs=[mask_state],
        outputs=[pixel_info]
    )

# 启动服务器
if __name__ == "__main__":
    # 如果是在本地运行，可以通过设置 server_name="0.0.0.0" 允许局域网其他设备访问
    interface.launch(share=False)