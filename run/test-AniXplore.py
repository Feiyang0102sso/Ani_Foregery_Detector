import os
import sys
import json
import time
import types
import inspect
import argparse
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import IMDLBenCo.training_scripts.utils.misc as misc

from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.datasets import AnimeDataset
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.evaluation import (PixelF1, ImageF1, PixelIOU, ImageAccuracy, PixelAccuracy, ImageAUC, PixelAUC, 
                                  ImageConfusionMatrix, SourceConfusionMatrix, PixelConfusionMatrix, PixelPrecision, PixelRecall, ImagePrecision, ImageRecall)
from IMDLBenCo.evaluation.Accuracy import SourceAccuracy
from IMDLBenCo.evaluation.F1 import SourcePrecision, SourceRecall, SourceF1

from IMDLBenCo.training_scripts.tester import test_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBench testing launch!', add_help=True)

    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--if_test_ImageF1', action='store_true')
    parser.add_argument('--if_test_ImageAccuracy', action='store_true')
    parser.add_argument('--if_test_PixelAccuracy', action='store_true')
    parser.add_argument('--if_test_ImageAUC', action='store_true')
    parser.add_argument('--if_test_PixelAUC', action='store_true')
    parser.add_argument('--if_test_PixelF1', action='store_true')
    parser.add_argument('--if_test_PixelIOU', action='store_true')
    parser.add_argument('--if_test_ImageConfusionMatrix', action='store_true')
    parser.add_argument('--if_test_PixelConfusionMatrix', action='store_true')

    # Add Precision/Recall metrics to align with train
    parser.add_argument('--if_test_PixelPrecision', action='store_true')
    parser.add_argument('--if_test_PixelRecall', action='store_true')
    parser.add_argument('--if_test_ImagePrecision', action='store_true')
    parser.add_argument('--if_test_ImageRecall', action='store_true')
    
    # Add Source metrics parameters
    parser.add_argument('--if_test_SourceAccuracy', action='store_true')
    parser.add_argument('--if_test_SourcePrecision', action='store_true')
    parser.add_argument('--if_test_SourceRecall', action='store_true')
    parser.add_argument('--if_test_SourceF1', action='store_true')
    parser.add_argument('--if_test_SourceConfusionMatrix', action='store_true')


    parser.add_argument('--raw_img_data_root', type=str)
    parser.add_argument('--edited_img_data_root', type=str)
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)
    
    # ----Dataset parameters----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--test_data_json', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset json, should be a json file contains many datasets. Details are in readme.md')
    # ------------------------------------
    # Testing parameters
    parser.add_argument('--checkpoint_path', default = '/root/workspace/IML-ViT/output_dir', type=str, help='path to the dir where saving checkpoints')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    parser.add_argument('--no_model_eval', action='store_true', 
                        help='Do not use model.eval() during testing.')

    # Log parameters-----------
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    # -----------------------
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # Add seeds for alignment
    parser.add_argument('--seed', default=42, type=int)

    args, remaining_args = parser.parse_known_args()


    model_class = MODELS.get(args.model)
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

def main(args, model_args):
    print("\nINTO test-anime-no-real (Single Card Mode) !!!!!!!!!!!!\n")
    
    # ================= Aligned: Output redirection =================
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(args.output_dir, "console_test_log.txt")
        sys.stdout = Logger(log_path, sys.stdout)
        sys.stderr = Logger(log_path, sys.stderr)
    # ===============================================================

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    # 修复：手动设置 distributed 属性为 False，因为这是单卡模式
    # tester.py 中会检查 args.distributed
    args.distributed = False
    
    # Aligned: seed for reproducibility
    misc.seed_torch(args.seed)

    test_transform = get_albu_transforms('test')

    # Aligned: Supporting directory for datasets (ConcatDataset)
    from torch.utils.data import ConcatDataset
    if os.path.isfile(args.test_data_json):
        dataset_test_all = AnimeDataset(
            args.test_data_json,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=None, # Will be set dataset by dataset if needed, but usually None for test
            raw_img_data_root=args.raw_img_data_root,
            edited_img_data_root=args.edited_img_data_root
        )
        test_dataset_dict = {Path(args.test_data_json).stem: dataset_test_all}
    else:
        test_dataset_dict = {}
        json_files = [f for f in os.listdir(args.test_data_json) if f.endswith('.json')]
        for json_file in json_files:
            dataset_name = Path(json_file).stem
            test_dataset_dict[dataset_name] = AnimeDataset(
                os.path.join(args.test_data_json, json_file),
                is_padding=args.if_padding,
                is_resizing=args.if_resizing,
                output_size=(args.image_size, args.image_size),
                common_transforms=test_transform,
                edge_width=args.edge_mask_width,
                post_funcs=None,
                raw_img_data_root=args.raw_img_data_root,
                edited_img_data_root=args.edited_img_data_root
            )
    
    global_rank = 0
    
    # ========define the model directly==========
    # model = IML_ViT(
    #     vit_pretrain_path = model_args.vit_pretrain_path,
    #     predict_head_norm= model_args.predict_head_norm,
    #     edge_lambda = model_args.edge_lambda
    # )
    
    # --------------- or -------------------------
    # Init model with registry
    model = MODELS.get(args.model)
    
    # Filt usefull args
    if isinstance(model,(types.FunctionType, types.MethodType)):
        model_init_params = inspect.signature(model).parameters
    else:
        model_init_params = inspect.signature(model.__init__).parameters
        
    combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    combined_args.update({k: v for k, v in vars(model_args).items() if k in model_init_params})
    model = model(**combined_args)
    # ============================================
    evaluator_list = [
        PixelF1(threshold=0.5, mode="origin"),
        PixelAccuracy(),
        PixelPrecision(),
        PixelRecall()
    ]
    if args.if_test_ImageF1:
        evaluator_list.append(ImageF1())
        print(f"add test ImageF1")
    if args.if_test_ImageAccuracy:
        evaluator_list.append(ImageAccuracy())
        print(f"add test ImageAccuracy")
    if args.if_test_PixelAccuracy:
        evaluator_list.append(PixelAccuracy())
        print(f"add test PixelAccuracy")
    if args.if_test_ImageAUC:
        evaluator_list.append(ImageAUC())
        print(f"add test ImageAUC")
    if args.if_test_PixelAUC:
        evaluator_list.append(PixelAUC())
        print(f"add test PixelAUC")
    if args.if_test_PixelF1:
        evaluator_list.append(PixelF1())
        print(f"add test PixelF1")
    if args.if_test_PixelIOU:
        evaluator_list.append(PixelIOU())
        print(f"add test PixelIOU")
    if args.if_test_ImageConfusionMatrix:
        evaluator_list.append(ImageConfusionMatrix())
        print(f"add test ImageConfusionMatrix")
    if args.if_test_PixelConfusionMatrix:
        evaluator_list.append(PixelConfusionMatrix())
        print(f"add test PixelConfusionMatrix")
    
    # Aligned metrics
    if args.if_test_PixelPrecision:
        evaluator_list.append(PixelPrecision())
        print(f"add test PixelPrecision")
    if args.if_test_PixelRecall:
        evaluator_list.append(PixelRecall())
        print(f"add test PixelRecall")
    if args.if_test_ImagePrecision:
        evaluator_list.append(ImagePrecision())
        print(f"add test ImagePrecision")
    if args.if_test_ImageRecall:
        evaluator_list.append(ImageRecall())
        print(f"add test ImageRecall")
        
    # Source metrics
    if args.if_test_SourceAccuracy:
        evaluator_list.append(SourceAccuracy())
        print(f"add test SourceAccuracy")
    if args.if_test_SourcePrecision:
        evaluator_list.append(SourcePrecision())
        print(f"add test SourcePrecision")
    if args.if_test_SourceRecall:
        evaluator_list.append(SourceRecall())
        print(f"add test SourceRecall")
    if args.if_test_SourceF1:
        evaluator_list.append(SourceF1())
        print(f"add test SourceF1")
    if args.if_test_SourceConfusionMatrix:
        evaluator_list.append(SourceConfusionMatrix())
        print(f"add test SourceConfusionMatrix")



    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    # Aligned: print parameter counts
    def count_parameters(model):
        for name, module in model.named_modules():
            if not list(module.parameters()):
                continue
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if len(name.split('.')) == 1:
                print(f"Module {name} has {param_count} trainable parameters.")
    count_parameters(model)
    
    start_time = time.time()
    # get post function (if have)
    post_function_name = f"{args.model}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None
    
    # Start go through each dataset:
    for dataset_name, dataset_test in test_dataset_dict.items():
        args.full_log_dir = os.path.join(args.log_dir, dataset_name)

        if global_rank == 0 and args.full_log_dir is not None:
            os.makedirs(args.full_log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.full_log_dir)
        else:
            log_writer = None
        
        # ------------------------------------
        print(dataset_test)
        print("len(dataset_test)", len(dataset_test))
        
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        print(f"Start testing on {dataset_name}! ")


        ckpt_path = args.checkpoint_path
        print(f"🔄 Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)
        
        # Aligned: Manual removal of mismatched parameters
        if 'model' in checkpoint:
            keys_to_remove = []
            for key in checkpoint['model']:
                if key.startswith(('auto_weight.', 'cls_head.', 'source_head.')):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del checkpoint['model'][key]
            if keys_to_remove:
                print(f"⚠️ Aligned: Removed incompatible old parameters from pretrained weights: {keys_to_remove}")

        model.load_state_dict(checkpoint['model'], strict=False)

        test_stats = test_one_epoch(
            model=model,
            data_loader=data_loader_test,
            evaluator_list=evaluator_list,
            device=device,
            epoch=args.epoch,
            log_writer=log_writer,
            args=args
        )
        log_stats = {
            **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': args.epoch}
    
        if args.full_log_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.full_log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        print(f"log_stats = {log_stats}")


        local_time = time.time() - start_time
        local_time_str = str(datetime.timedelta(seconds=int(local_time)))
        print(f'Testing on dataset {dataset_name} takes {local_time_str}')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total testing time {}'.format(total_time_str))
    exit(0)    
        


class Logger(object):
    def __init__(self, filename='console_log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
