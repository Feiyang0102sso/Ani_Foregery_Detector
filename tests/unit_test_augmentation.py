import os
import cv2
import shutil
import numpy as np

# 导入你的 transform
from IMDLBenCo.transforms.iml_transforms import get_albu_transforms

# =====================================================================
# 🎯 在这里填入你想指定的几张测试图片的绝对路径
TEST_IMAGES = [
    r"E:\!project dataset\fake_images\0000\inpainting\1013000\FLUX1_inpainting_1013000_tree.png",
    r"E:\!project dataset\fake_images\0000\inpainting\1013000\SD_inpainting_1013000_tree.png",
    r"E:\!project dataset\fake_images\0000\inpainting\1013000\SDXL_inpainting_1013000_tree.png"
]


# =====================================================================

def main():
    save_dir = "./specific_aug_results"
    os.makedirs(save_dir, exist_ok=True)

    # 获取我们设置了极端参数的 pipeline
    transform = get_albu_transforms('train_heavy')

    print("🔍 开始对指定图片进行残暴打击...")

    for idx, img_path in enumerate(TEST_IMAGES):
        if not os.path.exists(img_path):
            print(f"❌ 找不到图片，请检查路径: {img_path}")
            continue

        # 1. 使用 OpenCV 读取图片 (BGR 格式，最稳)
        image = cv2.imread(img_path)

        # 为了过 transform，捏造一个黑色的 Mask
        dummy_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 2. ⚡ 核心：执行数据增强！
        augmented = transform(image=image, mask=dummy_mask)

        # 取出处理后的 Numpy 图像数组 (0~255，直接就是图片格式)
        aug_img_numpy = augmented['image']

        # 3. 直接用 cv2 保存，不需要任何花里胡哨的转换
        base_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, f"AUG_{base_name}")
        cv2.imwrite(save_path, aug_img_numpy)

        print(f"✅ 已生成破坏后的图片: {save_path}")

    print(f"\n🎉 破坏完成！快去 {save_dir} 文件夹看看它歪了多少，糊成了什么样。")

    # ==========================================
    # 🔥 阅后即焚功能
    # ==========================================
    input("\n👉 验收完毕后，请按【Enter】键，将自动清空生成的测试图片...")

    try:
        shutil.rmtree(save_dir)
        print(f"🗑️ 成功！文件夹 {save_dir} 及其内容已被彻底清空。")
    except Exception as e:
        print(f"⚠️ 清理失败: {e}")


if __name__ == "__main__":
    main()