import os
import argparse
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_image(input_path, output_path, size):
    """
    读取单张图片，强行缩放为 size x size，并保存到指定路径。
    """
    try:
        with Image.open(input_path) as img:
            # 深度学习模型统一需要 3 通道 RGB 格式
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 暴力缩放为固定的正方形分辨率
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            # 提取小写后缀名用于判断保存格式
            ext = os.path.splitext(output_path)[1].lower()

            # 保存图片
            if ext in ['.jpg', '.jpeg']:
                img.save(output_path, format='JPEG', quality=95)
            elif ext == '.webp':
                img.save(output_path, format='WEBP', quality=95)
            else:
                img.save(output_path, format='PNG')
        return True

    except Exception as e:
        print(f"\n⚠️ 处理图片 {input_path} 时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="深度学习图片离线递归预处理工具")
    parser.add_argument('--folder_path', type=str, required=True, help='输入：原始图片总文件夹路径')
    parser.add_argument('--save_folder', type=str, default='', help='输出：总输出路径（默认原路径加 _r）')
    parser.add_argument('--size', type=int, default=512, help='目标尺寸：强制缩放为正方形 (默认: 512)')

    args = parser.parse_args()

    # 1. 自动处理总输出路径
    if not args.save_folder:
        clean_path = args.folder_path.rstrip('\\/')
        args.save_folder = f"{clean_path}_r"

    print("=========================================")
    print("🔍 正在扫描所有子文件夹...")

    # 2. 收集所有图片任务并构建对应的输出目录树
    tasks = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp')

    for root, _, files in os.walk(args.folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                # 图片的绝对物理路径
                input_path = os.path.join(root, file)

                # 获取该图片相对于输入根目录的“相对路径” (例如: "SDXL\img1.jpg")
                rel_path = os.path.relpath(input_path, args.folder_path)

                # 拼接到新的输出根目录上 (例如: "E:\wild_images_r\SDXL\img1.jpg")
                output_path = os.path.join(args.save_folder, rel_path)

                tasks.append((input_path, output_path))

    if not tasks:
        print("❌ 未在指定目录下找到任何支持的图片格式！")
        return

    print(f"📁 源总目录: {args.folder_path}")
    print(f"💾 输出总目录: {args.save_folder}")
    print(f"📐 目标尺寸: {args.size} x {args.size}")
    print(f"🚀 共扫描到图片: {len(tasks)} 张")
    print("=========================================")

    # 3. 提前创建好所有需要的子文件夹，避免多进程抢占创建时报错
    for _, output_path in tasks:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 4. 开启多进程加速处理
    success_count = 0
    max_workers = os.cpu_count() or 8

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_image, inp, out, args.size) for inp, out in tasks]

        # 显示进度条
        for future in tqdm(as_completed(futures), total=len(tasks), desc="图片缩放中"):
            if future.result():
                success_count += 1

    print(f"\n✅ 完美收工！成功缩放 {success_count} / {len(tasks)} 张图片。")
    print(f"👉 目录结构已完整克隆至: {args.save_folder}")


if __name__ == '__main__':
    main()