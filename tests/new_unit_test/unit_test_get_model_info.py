import os
import json
import torch
import tempfile
import unittest
import numpy as np
from PIL import Image

from IMDLBenCo import AnimeDataset, AnimeDatasetNoReal


# 请根据你的实际文件结构导入你的 Dataset 类
# from your_module.datasets import AnimeDataset, AnimeDatasetNoReal

def mock_img_loader(path):
    """
    模拟图片加载器，防止 AbstractDataset 在测试时因为找不到真实文件而报错。
    直接返回一个 10x10 的全黑 RGB PIL 图像。
    """
    return Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))


class TestAnimeDatasetGetItem(unittest.TestCase):
    def setUp(self):
        # 1. 准备你提供的可用测试数据
        self.test_data = [
            {
                "raw": "0000/1013000.jpg",
                "text": "0000/image_info/1013000.txt",
                "mask": "0000/mask/1013000/tree.png",
                "inpaint": [
                    "0000/inpainting/1013000/FLUX1_inpainting_1013000_tree.png",
                    "0000/inpainting/1013000/SD_inpainting_1013000_tree.png",
                    "0000/inpainting/1013000/SDXL_inpainting_1013000_tree.png"
                ],
                "t2i": [
                    "0000/text2image/1013000/SD_text2image_1013000.png",
                    "0000/text2image/1013000/SDXL_text2image_1013000.png",
                    "0000/text2image/1013000/FLUX1_text2image_1013000.png"
                ]
            },
            {
                "raw": "0000/148000.jpg",
                "text": "0000/image_info/148000.txt",
                "mask": None,
                "inpaint": None,
                "t2i": []
            }
        ]

        # 2. 创建一个临时 JSON 文件用于测试，测试结束会自动清理
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_path = os.path.join(self.temp_dir.name, 'test_data.json')
        with open(self.json_path, 'w') as f:
            json.dump(self.test_data, f)

        self.raw_root = "/dummy/raw"
        self.edited_root = "/dummy/edited"

    def tearDown(self):
        # 清理临时文件夹
        self.temp_dir.cleanup()

    def test_anime_dataset_get_item(self):
        """测试包含真实图片 (Real) 的 AnimeDataset"""
        # 传入 is_resizing=True 防止 AbstractDataset 抛出 AttributeError
        dataset = AnimeDataset(
            path=self.json_path,
            raw_img_data_root=self.raw_root,
            edited_img_data_root=self.edited_root,
            img_loader=mock_img_loader,
            is_resizing=True  # <--- 新增
        )

        expected_labels = [0, 1, 3, 2, 3, 2, 1, 0]
        self.assertEqual(len(dataset), 8, "AnimeDataset 应该包含 8 条数据记录")

        for idx, expected_label in enumerate(expected_labels):
            data_dict = dataset[idx]

            self.assertIn('source_label', data_dict, f"Index {idx} 缺少 'source_label' 键")
            self.assertEqual(
                data_dict['source_label'].item(),
                expected_label,
                f"Index {idx} 路径为 {dataset.tp_path[idx]}，预期 Label 为 {expected_label}，但得到 {data_dict['source_label'].item()}"
            )

    def test_anime_dataset_no_real_get_item(self):
        """测试排除真实图片的 AnimeDatasetNoReal"""
        # 同样传入 is_resizing=True
        dataset = AnimeDatasetNoReal(
            path=self.json_path,
            raw_img_data_root=self.raw_root,
            edited_img_data_root=self.edited_root,
            img_loader=mock_img_loader,
            is_resizing=True  # <--- 新增
        )

        expected_labels = [1, 3, 2, 3, 2, 1]
        self.assertEqual(len(dataset), 6, "AnimeDatasetNoReal 应该只包含 6 条生成/编辑记录")

        for idx, expected_label in enumerate(expected_labels):
            data_dict = dataset[idx]

            self.assertIn('source_label', data_dict, f"Index {idx} 缺少 'source_label' 键")
            self.assertEqual(
                data_dict['source_label'].item(),
                expected_label,
                f"Index {idx} 路径为 {dataset.tp_path[idx]}，预期 Label 为 {expected_label}，但得到 {data_dict['source_label'].item()}"
            )


if __name__ == '__main__':
    unittest.main()