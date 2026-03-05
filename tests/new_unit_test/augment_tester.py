import cv2
import numpy as np
import os
import random

# ==========================================
# ⚙️ 硬编码的常数配置
# ==========================================
# 根据你的报错截图，如果需要直接测试，可以将这里改成你的实际路径：
IMAGE_PATH = r"E:\!project dataset\fake_images\0000\inpainting\1013000\FLUX1_inpainting_1013000_tree.png"
# IMAGE_PATH = "test_image.jpg"


# ==========================================
# 🍃 温和的数据增强 (模拟日常平台操作)
# ==========================================

def aug_mild_jpeg_compression(image):
    """温和的JPEG压缩 (模拟社交平台上传压缩)"""
    # 将质量设置为 60 (满分 100)，会产生轻微的压缩伪影
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def aug_mild_blur(image):
    """轻微的高斯模糊 (模拟微小的失焦或平台缩放)"""
    return cv2.GaussianBlur(image, (5, 5), 0)


def aug_mild_brightness_contrast(image):
    """轻微的亮度和对比度调整"""
    alpha = random.uniform(0.9, 1.1)  # 对比度微调 (0.9 ~ 1.1)
    beta = random.randint(-15, 15)  # 亮度微调 (-15 ~ +15)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# ==========================================
# 😈 极端的奇怪增强 (原有的高强度破坏)
# ==========================================

def aug_heavy_noise(image):
    """强烈的随机高斯噪声"""
    row, col, ch = image.shape
    mean = 0
    var = 8000
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def aug_extreme_color_shift(image):
    """极度偏色 (修复了缺失 COLOR_ 前缀的 BUG)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(50, 150)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.5, 3.0), 0, 255)
    hsv = hsv.astype(np.uint8)
    # 🐛 Bug 修复: 改为 cv2.COLOR_HSV2BGR
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def aug_heavy_pixelate(image):
    """严重的马赛克/像素化"""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // 20), max(1, h // 20)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def aug_weird_rotation(image):
    """非直角的奇怪角度旋转"""
    h, w = image.shape[:2]
    angle = random.choice([47, 113, 226, 319])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, random.uniform(0.6, 1.2))
    return cv2.warpAffine(image, M, (w, h))


def aug_mixed_hell(image):
    """混合地狱增强"""
    img = aug_weird_rotation(image)
    img = aug_extreme_color_shift(img)
    img = aug_heavy_noise(img)
    return img


# ==========================================
# 🚀 主运行逻辑
# ==========================================

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image not found -> {IMAGE_PATH}")
        return

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ Error: Cannot decode image -> {IMAGE_PATH}")
        return

    base_name, ext = os.path.splitext(IMAGE_PATH)

    # 注册所有的增强操作，现在后缀全部换成英文了
    augmentations = {
        "mild_jpeg": aug_mild_jpeg_compression,
        "mild_blur": aug_mild_blur,
        "mild_bright_contrast": aug_mild_brightness_contrast,
        "heavy_noise": aug_heavy_noise,
        "extreme_color_shift": aug_extreme_color_shift,
        "heavy_pixelate": aug_heavy_pixelate,
        "weird_rotation": aug_weird_rotation,
        "mixed_hell": aug_mixed_hell
    }

    generated_files = []

    print(f"✅ Loaded: {IMAGE_PATH}")
    print("-" * 40)
    print("Generating augmented images...\n")

    # 执行并保存
    for suffix_name, aug_func in augmentations.items():
        result_img = aug_func(img.copy())

        out_path = f"{base_name}_{suffix_name}{ext}"
        cv2.imwrite(out_path, result_img)
        generated_files.append(out_path)
        print(f"📸 Generated: {out_path}")

    print("-" * 40)
    print(f"🎉 Done! {len(augmentations)} images created.")
    print("-" * 40)

    # 等待用户查看后按下 Enter
    input("⚠️ Press [Enter] to delete all generated images and clean up...")

    # 自动清理
    print("\nCleaning up...")
    for file_path in generated_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Deleted: {file_path}")

    print("✨ All clean! Exiting.")


if __name__ == "__main__":
    main()