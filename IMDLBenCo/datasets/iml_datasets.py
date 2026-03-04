import json
import os


from ..registry import DATASETS
from .abstract_dataset import AbstractDataset

from tqdm import tqdm
import itertools
import shutil

@DATASETS.register_module()
class ManiDataset(AbstractDataset):
    def _init_dataset_path(self, path):
        self.entry_path = path
        tp_dir = os.path.join(path, 'Tp')
        gt_dir = os.path.join(path, 'Gt')
        
        assert os.path.isdir(path), NotADirectoryError(f"Get Error when loading from {self.entry_path}, the path is not a directory. Please check the path.")
        assert os.path.isdir(tp_dir), NotADirectoryError(f"Get Error when loading from {tp_dir}, the Tp directory is not exist. Please check the path.")
        assert os.path.isdir(gt_dir), NotADirectoryError(f"Get Error when loading from {gt_dir}, the Gt directory is not exist. Please check the path.")
        
        tp_list = os.listdir(tp_dir)
        gt_list = os.listdir(gt_dir)
        # Use sort mathod to keep order, to make sure the order is the same as the order in the tp_list and gt_list
        tp_list.sort()
        gt_list.sort()
        t_tp_list = [os.path.join(path, 'Tp', tp_list[index]) for index in range(len(tp_list))]
        t_gt_list = [os.path.join(path, 'Gt', gt_list[index]) for index in range(len(gt_list))]
        return t_tp_list, t_gt_list
    
@DATASETS.register_module()
class JsonDataset(AbstractDataset):
    """ init from a json file, which contains all the images path
        file is organized as:
            [
                ["./Tp/6.jpg", "./Gt/6.jpg"],
                ["./Tp/7.jpg", "./Gt/7.jpg"],
                ["./Tp/8.jpg", "Negative"],
                ......
            ]
        if path is "Neagative" then the image is negative sample, which means ground truths is a totally black image, and its label == 0.
        
    Args:
        path (_type_): _description_
        transform_albu (_type_, optional): _description_. Defaults to None.
        mask_edge_generator (_type_, optional): _description_. Defaults to None.
        if_return_shape
    """
    def _init_dataset_path(self, path):
        self.entry_path = path
        try:
            images = json.load(open(path, 'r'))
        except:
            raise TypeError(f"Get Error when loading from {self.entry_path}, please check the file format, it should be a json file, and the content should be like: [['./Tp/6.jpg', './Gt/6.jpg'], ['./Tp/7.jpg', './Gt/7.jpg'], ['./Tp/8.jpg', 'Negative'], ......]")
        tp_list = []
        gt_list = []
        for record in images:
            if os.path.isfile(record[0]):
                tp_list.append(record[0])
                gt_list.append(record[1])
            else: 
                raise TypeError(f"Get Error when loading from {self.entry_path}, the error record is: {record[0]}, which is not a file. Please check this file or try to use absolute path instead. Otherwise if you want to use ManiDataset with a path instead of JsonDataset, please pass a path into the 'train_*.sh'. For more information please see the protocol here: https://scu-zjz.github.io/IMDLBenCo-doc/guide/quickstart/0_dataprepare.html#specific-format-definitions")
        return tp_list, gt_list

# 用于训练 包含所有图片 包括真实图片
@DATASETS.register_module()
class AnimeDataset(AbstractDataset):
    """ a dir contains many json file
        file is organized as:
        [
            {
                "raw": "0000/5468000.jpg",
                "text": "0000/image_info/5468000.txt",
                "mask": "0000/mask/5468000",
                "inpaint": "0000/inpainting/5468000",
                "t2i": "0000/text2image/5468000"
            },
            {
                "raw": "0000/547000.png",
                "text": "0000/image_info/547000.txt",
                "mask": "0000/mask/547000",
                "inpaint": null,
                "t2i": null
            }
        ]
    """

    def _init_dataset_path(self, path):
        self.entry_path = path
        # print(f"得到path = {path}")
        try:
            data_descriptions = []
            if os.path.isfile(path):
                initial_data = json.load(open(path, 'r'))
                if isinstance(initial_data, list):
                    data_descriptions.append(initial_data)
                elif isinstance(initial_data, dict):
                    print(f"[Info] Detected index file: {path}")
                    for key, file_path in initial_data.items():
                        if os.path.isfile(file_path):
                            print(f"Loading data from referenced file: {file_path}")
                            loaded_content = json.load(open(file_path, 'r'))
                            if isinstance(loaded_content, list):
                                data_descriptions.append(loaded_content)
                            else:
                                print(f"[Warning] Content of {file_path} is not a list, skipping.")
                        else:
                            print(f"[Warning] Referenced file not found: {file_path}")
                else:
                    # pass
                    raise TypeError(f"Unexpected JSON format in {path}. Expected list or dict.")
            else:
                json_files = os.listdir(path)
                # print(f"json_files = {json_files}")
                for json_file in tqdm(json_files):
                    data_descriptions.append(json.load(open(os.path.join(path, json_file), 'r')))
                print("Loaded all json files")
        except Exception as e:
            raise TypeError(f"Get Error when loading from {path}: {e}")

        tp_list = []
        gt_list = []
        #print("Finish reading json")
        all_records = list(itertools.chain.from_iterable(data_descriptions))
        #print(f"共有 {len(all_records)} 条record")

        # 使用 tqdm 遍历扁平化后的记录
        for record in all_records:

            # Add raw image
            if record['raw'] != 'null' and record['raw'] != None:
                raw_img_path = os.path.join(self.raw_img_data_root, record['raw'])
                tp_list.append(raw_img_path)
                gt_list.append('raw')
            else:
                pass
                # print(f"[Exception] {record} raw is None")

            # Add inapinting image
            if record['inpaint'] != 'null' and record['inpaint'] != None:
                inpaint_imgs = [os.path.join(self.edited_img_data_root, img) for img in record["inpaint"]]
                tp_list.extend(inpaint_imgs)
            
                mask_path = [os.path.join(self.edited_img_data_root, record["mask"])] *len(inpaint_imgs)
                gt_list.extend(mask_path)
            # Add t2i image
            if record['t2i'] != 'null' and record['t2i'] != None:
                # t2i_imgs = os.listdir(os.path.join(self.edited_img_data_root, record['t2i']))
                t2i_imgs = [os.path.join(self.edited_img_data_root, img) for img in record["t2i"]]
                # for t2i_img in t2i_imgs:
                    # t2i_img_path = os.path.join(self.edited_img_data_root, record['t2i'], t2i_img)
                tp_list.extend(t2i_imgs)
                gt_list.extend(['text2img']*len(t2i_imgs))
            # else:
                #print(f"[Info] {record} t2i is None")
                # pass
        assert len(tp_list) == len(gt_list), print("{} != {}".format(len(tp_list), len(gt_list)))
        
        return tp_list, gt_list

    # add new
    def __getitem__(self, index, num_trials=3):
        data_dict = super().__getitem__(index)

        img_path = self.tp_path[index]

        # 1: FLUX, 2: SDXL, 3: SD, 0: Real
        if 'FLUX' in img_path:
            source_label = 1
            # print(f"AnimeDataset: {img_path} -> Label: {source_label}")
        elif any(kw in img_path for kw in ['SDXL', 'Pony', 'Illustrious']):
            source_label = 2
            # print(f"AnimeDataset: {img_path} -> Label: {source_label}")
        elif 'SD' in img_path:
            source_label = 3
            # print(f"AnimeDataset: {img_path} -> Label: {source_label}")
        else:
            source_label = 0
            # print(f"AnimeDataset: {img_path} -> Label: {source_label}")

        import torch
        data_dict['source_label'] = torch.tensor(source_label, dtype=torch.long)

        return data_dict

# 用于测试 排除真的图片那因为他们的f1 score是0
@DATASETS.register_module()
class AnimeDatasetNoReal(AbstractDataset):
    """ a dir contains many json file
        file is organized as:
        [
            {
                "raw": "0000/5468000.jpg",
                "text": "0000/image_info/5468000.txt",
                "mask": "0000/mask/5468000",
                "inpaint": "0000/inpainting/5468000",
                "t2i": "0000/text2image/5468000"
            },
            {
                "raw": "0000/547000.png",
                "text": "0000/image_info/547000.txt",
                "mask": "0000/mask/547000",
                "inpaint": null,
                "t2i": null
            }
        ]
    """

    def _init_dataset_path(self, path):
        self.entry_path = path
        # print(f"得到path = {path}")
        try:
            data_descriptions = []
            if os.path.isfile(path):
                initial_data = json.load(open(path, 'r'))
                if isinstance(initial_data, list):
                    data_descriptions.append(initial_data)
                elif isinstance(initial_data, dict):
                    print(f"[Info] Detected index file: {path}")
                    for key, file_path in initial_data.items():
                        if os.path.isfile(file_path):
                            print(f"Loading data from referenced file: {file_path}")
                            loaded_content = json.load(open(file_path, 'r'))
                            if isinstance(loaded_content, list):
                                data_descriptions.append(loaded_content)
                            else:
                                print(f"[Warning] Content of {file_path} is not a list, skipping.")
                        else:
                            print(f"[Warning] Referenced file not found: {file_path}")
                else:
                    raise TypeError(f"Unexpected JSON format in {path}. Expected list or dict.")
            else:
                json_files = os.listdir(path)
                # print(f"json_files = {json_files}")
                for json_file in tqdm(json_files):
                    data_descriptions.append(json.load(open(os.path.join(path, json_file), 'r')))
                print("Loaded all json files")
        except Exception as e:
            raise TypeError(f"Get Error when loading from {path}: {e}")

        tp_list = []
        gt_list = []
        #print("Finish reading json")
        all_records = list(itertools.chain.from_iterable(data_descriptions))
        #print(f"共有 {len(all_records)} 条record")

        # 使用 tqdm 遍历扁平化后的记录
        for record in all_records:

            # 不考虑真实图片
            # Add raw image
            # if record['raw'] != 'null' and record['raw'] != None:
            #     raw_img_path = os.path.join(self.raw_img_data_root, record['raw'])
            #     tp_list.append(raw_img_path)
            #     gt_list.append('raw')
            # else:
            #     print(f"[Exception] {record} raw is None")

            # Add inapinting image
            if record['inpaint'] != 'null' and record['inpaint'] != None:
                inpaint_imgs = [os.path.join(self.edited_img_data_root, img) for img in record["inpaint"]]
                tp_list.extend(inpaint_imgs)
            
                mask_path = [os.path.join(self.edited_img_data_root, record["mask"])] *len(inpaint_imgs)
                gt_list.extend(mask_path)
            # Add t2i image
            if record['t2i'] != 'null' and record['t2i'] != None:
                # t2i_imgs = os.listdir(os.path.join(self.edited_img_data_root, record['t2i']))
                t2i_imgs = [os.path.join(self.edited_img_data_root, img) for img in record["t2i"]]
                # for t2i_img in t2i_imgs:
                    # t2i_img_path = os.path.join(self.edited_img_data_root, record['t2i'], t2i_img)
                tp_list.extend(t2i_imgs)
                gt_list.extend(['text2img']*len(t2i_imgs))
            # else:
                #print(f"[Info] {record} t2i is None")
                # pass
        assert len(tp_list) == len(gt_list), print("{} != {}".format(len(tp_list), len(gt_list)))
        
        return tp_list, gt_list

    def __getitem__(self, index, num_trials=3):
        data_dict = super().__getitem__(index)

        img_path = self.tp_path[index]
        # print("AnimeDatasetNoReal: " + img_path)

        # 1: FLUX, 2: SDXL, 3: SD, 0: Real
        if 'FLUX' in img_path:
            source_label = 1
            # print(f"AnimeDatasetNoReal: {img_path} -> Label: {source_label}")
        elif any(kw in img_path for kw in ['SDXL', 'Pony', 'Illustrious']):
            source_label = 2
            # print(f"AnimeDatasetNoReal: {img_path} -> Label: {source_label}")
        elif 'SD' in img_path:
            source_label = 3
            # print(f"AnimeDatasetNoReal: {img_path} -> Label: {source_label}")
        else:
            source_label = 0
            # print(f"AnimeDatasetNoReal: {img_path} -> Label: {source_label}")

        import torch
        data_dict['source_label'] = torch.tensor(source_label, dtype=torch.long)

        return data_dict
    

@DATASETS.register_module()
class CivitAI(AbstractDataset):
    def _init_dataset_path(self, path):
        self.entry_path = path
        tp_list = []
        gt_list = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                tp_list.append(os.path.join(self.edited_img_data_root, line.strip()))
                gt_list.append("text2img")
        assert len(tp_list) == len(gt_list), print("{} != {}".format(len(tp_list), len(gt_list)))
        return tp_list, gt_list