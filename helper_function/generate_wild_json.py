import json
import random
import math
from pathlib import Path
from PIL import Image

# ================== 配置区域 ==================

# 真实图片 基准目录 与 指定读取的子包
REAL_BASE_DIR = Path(r"E:\!project dataset\real_images")
REAL_SUB_DIR = "0001"
REAL_ROOT = REAL_BASE_DIR / REAL_SUB_DIR

# AI 图片主目录 (脚本会自动读取其下的 SD, SDXL, FLUX 等子文件夹)
AI_BASE_DIR = Path(r"E:\!project dataset\fake_images")
AI_SUB_DIR = "wild_imgs"
AI_ROOT = AI_BASE_DIR / AI_SUB_DIR

# --- 物理拆分的目标文件夹配置 ---
REAL_TRAIN_DIR = REAL_BASE_DIR / f"{REAL_SUB_DIR}_train"
REAL_TEST_DIR = REAL_BASE_DIR / f"{REAL_SUB_DIR}_test"

AI_TRAIN_DIR = AI_BASE_DIR / f"{AI_SUB_DIR}_train"
AI_TEST_DIR = AI_BASE_DIR / f"{AI_SUB_DIR}_test"

# JSON 输出位置 (依然存入 run 文件夹)
OUTPUT_DIR = Path("out")
OUTPUT_TRAIN = OUTPUT_DIR / "wild_train.json"
OUTPUT_TEST = OUTPUT_DIR / "wild_test.json"

# 训练/测试划分
TRAIN_RATIO = 0.8  # 8:2

# 真实 : AI 的整体比例（1:3 即 1张真实配3张AI）
REAL_TO_AI_RATIO = 1.0 / 3.0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --- 新增：统一 Resize 的分辨率常量 ---
TARGET_SIZE = (512, 512)


# ============================================

def list_images(root: Path):
    """只在传入的 root 及其子目录中搜索图片"""
    return [
        p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]


def make_real_record(rel_path: str):
    return {"raw": rel_path, "text": None, "mask": None, "inpaint": None, "t2i": None}


def make_ai_record(rel_path: str):
    return {"raw": None, "text": None, "mask": None, "inpaint": None, "t2i": [rel_path]}


def is_corrupted_image(img_path: Path):
    """
    检查图片是否损坏。
    返回: (is_corrupt: bool, error_message: str)
    """
    try:
        with Image.open(img_path) as img:
            img.verify()  # 检查图像文件的完整性，不完全加载进内存，速度较快
        return False, ""
    except Exception as e:
        return True, str(e)


def resize_and_get_rel(src_path: Path, src_root: Path, dest_root: Path, base_dir: Path):
    """
    负责把源文件读取、Resize，保存到新文件夹，并返回用于 JSON 的相对路径 (相对于 base_dir)
    """
    rel_to_root = src_path.relative_to(src_root)
    dest_path = dest_root / rel_to_root

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 执行图片读取、Resize 和保存
    with Image.open(src_path) as img:
        # 如果是带有透明通道的格式，但在保存为 JPEG 时会报错，需要转换为 RGB
        if dest_path.suffix.lower() in {".jpg", ".jpeg"} and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # 采用 LANCZOS 高质量重采样算法进行 Resize
        resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        resized_img.save(dest_path)

    return dest_path.relative_to(base_dir).as_posix()


def process_and_copy(src_list, src_root, dest_root, base_dir, record_maker, records_list, corrupt_records):
    """
    遍历列表，检查图片，若正常则 Resize 并生成记录；若损坏则跳过并记录。
    """
    for p in src_list:
        # 1. 检测图片是否损坏
        is_corrupt, err_msg = is_corrupted_image(p)
        if is_corrupt:
            print(f"⚠️ 发现损坏的图片并跳过: {p}")
            corrupt_records.append({"path": str(p), "error": err_msg})
            continue

        # 2. 正常图片执行 Resize 与路径记录
        try:
            rel = resize_and_get_rel(p, src_root, dest_root, base_dir)
            records_list.append(record_maker(rel))
        except Exception as e:
            # 捕获 verify 阶段无法识别，但在实际读取处理时报错的深层损坏文件
            print(f"⚠️ 处理图片时发生错误并跳过: {p}")
            corrupt_records.append({"path": str(p), "error": f"Resize Error: {str(e)}"})


def main():
    # 1. 收集真实图片
    real_imgs = list_images(REAL_ROOT)
    print(f"✅ Found {len(real_imgs)} real images strictly in '{REAL_ROOT.name}'")

    # 2. 分类收集 AI 图片
    ai_categories = {}
    total_ai_imgs = 0

    for sub_dir in AI_ROOT.iterdir():
        if sub_dir.is_dir():
            imgs = list_images(sub_dir)
            if imgs:
                ai_categories[sub_dir.name] = imgs
                total_ai_imgs += len(imgs)
                print(f"✅ Found {len(imgs)} AI images in '{sub_dir.name}'")

    if not real_imgs or total_ai_imgs == 0:
        raise RuntimeError("❌ 真实图或 AI 图数量为 0，请检查路径。")

    # 3. 计算配额
    max_real = len(real_imgs)
    max_ai = int(max_real / REAL_TO_AI_RATIO)

    if max_ai > total_ai_imgs:
        max_ai = total_ai_imgs
        max_real = int(max_ai * REAL_TO_AI_RATIO)

    # 4. 抽取并打乱真实图片
    random.shuffle(real_imgs)
    real_used = real_imgs[:max_real]

    # 5. 平衡抽取 AI 图片
    ai_used = []
    cat_names = list(ai_categories.keys())
    cat_names.sort(key=lambda c: len(ai_categories[c]))  # 优先分发数量少的

    remaining_target = max_ai
    remaining_cats = len(cat_names)

    print("\n--- ⚖️ 平衡采样分配结果 ---")
    for cat in cat_names:
        quota = math.ceil(remaining_target / remaining_cats)
        available = len(ai_categories[cat])
        take = min(quota, available)

        random.shuffle(ai_categories[cat])
        ai_used.extend(ai_categories[cat][:take])

        remaining_target -= take
        remaining_cats -= 1
        print(f"Model '{cat}': Expected {quota} -> Took {take} images.")

    # 6. 将抽取的数据先划分为 Train 和 Test 集合
    n_real_train = int(len(real_used) * TRAIN_RATIO)
    real_train_src = real_used[:n_real_train]
    real_test_src = real_used[n_real_train:]

    random.shuffle(ai_used)  # 打乱防止某个模型全扎堆在某个切片
    n_ai_train = int(len(ai_used) * TRAIN_RATIO)
    ai_train_src = ai_used[:n_ai_train]
    ai_test_src = ai_used[n_ai_train:]

    # 7. 物理读取、Resize、验证损坏并生成 JSON Record
    print(f"\n🚚 开始处理文件，目标分辨率 {TARGET_SIZE} (包含 Resize 等耗时操作，请耐心等待)...")
    train_records = []
    test_records = []
    corrupt_records = []  # 用于记录所有损坏的图片信息

    # 复制 Train 数据
    process_and_copy(real_train_src, REAL_ROOT, REAL_TRAIN_DIR, REAL_BASE_DIR, make_real_record, train_records,
                     corrupt_records)
    process_and_copy(ai_train_src, AI_ROOT, AI_TRAIN_DIR, AI_BASE_DIR, make_ai_record, train_records, corrupt_records)

    # 复制 Test 数据
    process_and_copy(real_test_src, REAL_ROOT, REAL_TEST_DIR, REAL_BASE_DIR, make_real_record, test_records,
                     corrupt_records)
    process_and_copy(ai_test_src, AI_ROOT, AI_TEST_DIR, AI_BASE_DIR, make_ai_record, test_records, corrupt_records)

    print("✅ 处理与检测完成！")

    # 8. 最后再打乱一次记录，避免训练时出现集中性抖动
    random.shuffle(train_records)
    random.shuffle(test_records)

    # 9. 导出 JSON
    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEST.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_TRAIN.open("w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with OUTPUT_TEST.open("w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    print(f"\n📂 已保存 {len(train_records)} train records to {OUTPUT_TRAIN}")
    print(f"📂 已保存 {len(test_records)} test records to {OUTPUT_TEST}")

    # 10. 统一汇报坏图结果
    if len(corrupt_records) > 0:
        print(f"\n❌ 共遇到 {len(corrupt_records)} 张 Corrupt 损坏/处理失败的图片:")
        for record in corrupt_records:
            print(f"   - 路径: {record['path']}  |  信息: {record['error']}")
    else:
        print("\n🎉 完美！本次运行未发现任何损坏的图片。")


if __name__ == "__main__":
    main()