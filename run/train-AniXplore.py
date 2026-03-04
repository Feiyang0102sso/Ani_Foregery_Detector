import os
import sys
import json
import time
import types
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path

from tqdm import tqdm
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.evaluation.Accuracy import SourceAccuracy
from IMDLBenCo.evaluation.F1 import SourcePrecision, SourceRecall, SourceF1

from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.transforms import get_albu_transforms

from IMDLBenCo.datasets import ManiDataset, JsonDataset, BalancedDataset
from IMDLBenCo.datasets import AnimeDataset, AnimeDatasetNoReal
from IMDLBenCo.evaluation import (PixelF1, ImageF1, PixelIOU, ImageAccuracy, PixelAccuracy, ImageAUC, PixelAUC,
                                  PixelPrecision, PixelRecall, ImagePrecision, ImageRecall)

from IMDLBenCo.training_scripts.tester import test_one_epoch
from IMDLBenCo.training_scripts.trainer import train_one_epoch


import warnings

warnings.filterwarnings("ignore")

def count_parameters(model):
    for name, module in model.named_modules():
        if not list(module.parameters()):
            continue
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if len(name.split('.')) == 1:
            print(f"Module {name} has {param_count} trainable parameters.")


def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBenCo training launch!', add_help=True)

    # Model name
    parser.add_argument('--if_test_ImageF1', action='store_true')
    parser.add_argument('--if_test_ImageAccuracy', action='store_true')
    parser.add_argument('--if_test_PixelAccuracy', action='store_true')
    parser.add_argument('--if_test_ImageAUC', action='store_true')
    parser.add_argument('--if_test_PixelAUC', action='store_true')
    parser.add_argument('--if_test_PixelF1', action='store_true')
    parser.add_argument('--if_test_PixelIOU', action='store_true')

    # add new
    parser.add_argument('--if_test_PixelPrecision', action='store_true')
    parser.add_argument('--if_test_PixelRecall', action='store_true')
    parser.add_argument('--if_test_ImagePrecision', action='store_true')
    parser.add_argument('--if_test_ImageRecall', action='store_true')

    parser.add_argument('--resume',  default=None, type=str, help="resume from the latest ckpt")
    parser.add_argument('--raw_img_data_root', type=str)
    parser.add_argument('--edited_img_data_root', type=str)
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)
    
    parser.add_argument('--if_predict_label', action='store_true',
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    # ----Dataset parameters ----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--data_path', default='/root/Dataset/CASIA2.0/', type=str,
                        help='dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    # ------------------------------------
    # training related
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    parser.add_argument('--epochs', default=200, type=int)
    # Test related
    parser.add_argument('--no_model_eval', action='store_true', 
                        help='Do not use model.eval() during testing.')
    parser.add_argument('--test_period', default=1, type=int,
                        help="how many epoch per testing one time")
    
    # Output interval
    parser.add_argument('--log_per_epoch_count', default=20, type=int,
                        help="how many loggings (data points for loss) per testing epoch in Tensorboard")
    
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='find_unused_parameters for DDP. Mainly solve issue for model with image-level prediction but not activate during training.')
    
    # If AMP
    parser.add_argument('--if_not_amp', action='store_false',
                        help='Do not use automatic precision.')
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # ----Log parameters-----------
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    # -----------------------
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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
    
    parser.add_argument('--checkpoint_path', type=str, default="", help="pretrained ckpt path")

    # 只让 source_head 和 auto_weight 训练，其它所有参数都冻结。
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='If enabled, all network parameters except source_head and auto_weight are frozen. '
                             'This is typically used when a new classification head is first added.')

    # 只冻结 source_head 和 auto_weight，其它所有参数都可以训练
    parser.add_argument('--freeze_source', action='store_true',
                        help='If enabled, source_head and auto_weight will be frozen to preserve current source tracing ability.')

    # 针对野图微调：只训练图像级判别头（保持像素级 inpainting 能力完全不动）
    parser.add_argument(
        '--wild_cls_only',
        action='store_true',
        help='When set, only the image-level classification head is trainable during finetuning on wild images. '
             'All mask-related branches (fusion_layers, featurePyramid_net, predict_head) remain frozen so that '
             'pixel-wise inpainting localization stays unchanged.'
    )

    parser.add_argument('--use_heavy_aug', action='store_true',
                        help='Training with data augmentation')

    # 针对野图微调或消融实验：不计算/不反向传播溯源源（来源模型）分类损失
    parser.add_argument('--disable_source_loss', action='store_true',
                        help='If enabled, disables the source tracing loss calculation during training. Useful for wild image finetuning where source labes are unreliable or nonexistent.')

    args, remaining_args = parser.parse_known_args()

    model_class = MODELS.get(args.model)

    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args



def main(args, model_args):

    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    # ================= 新增：接管系统输出 =================
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(args.output_dir, "console_log.txt")
        sys.stdout = Logger(log_path, sys.stdout)
        sys.stderr = Logger(log_path, sys.stderr)  # 把报错信息也一起存进去
    # ======================================================

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # train_transform = get_albu_transforms('train')
    # ==== whether start data Augmentation ====
    if args.use_heavy_aug:
        print("🌪️ [Augmentation Enabled] Heavy Augmentation is now on")
        train_transform = get_albu_transforms('train_heavy',)
    else:
        print("🌱 [Augmentation Disabled] ")
        train_transform = get_albu_transforms('train')

    test_transform = get_albu_transforms('test')

    # get post function (if have)
    post_function_name = f"{args.model}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None



    from torch.utils.data import ConcatDataset

    if os.path.isfile(args.data_path):
        dataset_train = AnimeDataset(
            args.data_path,    
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=train_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function,
            raw_img_data_root=args.raw_img_data_root,
            edited_img_data_root=args.edited_img_data_root
        )
    else:
        dataset_train_list = []
        json_files = os.listdir(args.data_path)
        for json_file in json_files:
            dataset_train_list.append(
                AnimeDataset(
                    os.path.join(args.data_path, json_file),    
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=train_transform,
                    edge_width=args.edge_mask_width,
                    post_funcs=post_function,
                    raw_img_data_root=args.raw_img_data_root,
                    edited_img_data_root=args.edited_img_data_root
                )
            )
        dataset_train = ConcatDataset(dataset_train_list)
        print(f"[info] concat finish")


    test_datasets = {}

    if os.path.isfile(args.test_data_path):
        dataset_test = AnimeDatasetNoReal(
            args.test_data_path,
            is_padding=args.if_padding,
            is_resizing=args.if_resizing,
            output_size=(args.image_size, args.image_size),
            common_transforms=test_transform,
            edge_width=args.edge_mask_width,
            post_funcs=post_function,
            raw_img_data_root=args.raw_img_data_root,
            edited_img_data_root=args.edited_img_data_root
        )
    else:
        dataset_test_list = []
        json_files = os.listdir(args.test_data_path)
        for json_file in json_files:
            dataset_test_list.append(
                AnimeDatasetNoReal(
                    os.path.join(args.test_data_path, json_file),
                    is_padding=args.if_padding,
                    is_resizing=args.if_resizing,
                    output_size=(args.image_size, args.image_size),
                    common_transforms=test_transform,
                    edge_width=args.edge_mask_width,
                    post_funcs=post_function,
                    raw_img_data_root=args.raw_img_data_root,
                    edited_img_data_root=args.edited_img_data_root,
                    # weights_only=False
                )
            )
        dataset_test = ConcatDataset(dataset_test_list)
        print(f"[info] concat finish")
    
    test_datasets['test_dataset'] = dataset_test
    


    # ------------------------------------
    print(dataset_train)
    print(test_datasets)
    test_sampler = {}
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        for name, dataset_test in test_datasets.items():
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, 
                num_replicas=num_tasks, 
                rank=global_rank, 
                shuffle=False,
                drop_last=True
            )
            test_sampler[name] = sampler_test
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_test = %s" % str(sampler_test))
    else:
        # 修改为0 进行初始化
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        for name,dataset_test in test_datasets.items():
            sampler_test = torch.utils.data.RandomSampler(dataset_test)
            test_sampler[name] = sampler_test
    
    if global_rank == 0:
        # import wandb
        # import socket
        # # Get the hostname
        # host_name = socket.gethostname()
        # project_name = 'AnimeDL2M-' + host_name
        # wandb.init(project=project_name)
        pass

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    test_data_loader = {}
    for name in test_sampler.keys():
        data_loader_test = torch.utils.data.DataLoader(
            test_datasets[name], sampler=test_sampler[name],
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        test_data_loader[name] = data_loader_test
    

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
        PixelRecall(),
        SourceAccuracy(),
        SourcePrecision(),
        SourceRecall(),
        SourceF1()
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

        # ========== add new args ==========
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
        # ====================================

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    
    if args.checkpoint_path != "" or args.resume != None:
        # from the latest ckpt
        saved_ckpt_files = [possible_ckpt for possible_ckpt in os.listdir(args.output_dir) if possible_ckpt.endswith(".pth")]
        if args.resume != None and len(saved_ckpt_files) > 0:
            saved_ckpt_pairs = [(int(saved_ckpt_file.split('-')[1].split('.')[0]) , saved_ckpt_file) for saved_ckpt_file in saved_ckpt_files]
            saved_ckpt_pairs.sort(key=lambda x: x[0], reverse=True)
            ckpt_epoch, ckpt_name = saved_ckpt_pairs[0]
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            args.start_epoch = ckpt_epoch + 1
        else:
            # from pretrained ckpt
            ckpt_path = args.checkpoint_path
            args.start_epoch = 0

        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)

        # ----手动删除形状不匹配的参数 ----
        if 'model' in checkpoint:
            keys_to_remove = []
            for key in checkpoint['model']:
                # auto_weight.params: old shape may differ
                # cls_head / source_head: architecture changed (adapter-based now)
                # cls_adapter: doesn't exist in old checkpoints (will be fresh)
                if key.startswith(('auto_weight.', 'cls_head.', 'source_head.')):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del checkpoint['model'][key]
            if keys_to_remove:
                print(f"⚠️ 从预训练权重中剔除不兼容的旧参数: {keys_to_remove}")

        # 允许模型加载时“忽略不匹配的权重”，这样才能保留旧知识的同时学习新器官！
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint['model'], strict=False)

        # =================================================================
        # 🔥 野图微调专属策略：冻结重型骨干与溯源头，仅训练检测相关模块
        # =================================================================
        print("=> 实施【野图伪造检测微调】策略：冻结骨干网络，切断溯源分支...")

        # 1. 第一步：残暴地把所有参数全部冻结
        for param in model.parameters():
            param.requires_grad = False

        # 2. 第二步：精准解冻负责“伪造检测”的核心模块
        # trainable_modules = ['cls_head',
        #                      'predict_head',
        #                      'featurePyramid_net',
        #                      'fusion_layers',
        #                      # 'auto_weight',  #不再训练 auto_weight，因为已经硬编码了 Loss
        #                      'convnext.stages.2',
        #                      'segformer.block4']

        # 2. 第二步：精准解冻需要训练的模块
        #    默认策略：同时微调图像级分类头 + 像素级掩码分支（适合全监督场景）
        #    野图微调（--wild_cls_only）：只训练图像级分类头，完全冻结掩码分支，确保 inpainting 能力不被破坏
        if args.wild_cls_only:
            print("🎯 [Wild Finetune] Enable wild_cls_only: "
                  "freeze mask branch (fusion_layers / featurePyramid_net / predict_head), "
                  "train cls_adapter + cls_head + source_head on wild images.")
            trainable_modules = [
                'cls_adapter',
                'cls_head',
                'source_head',
                'auto_weight'
            ]
        else:
            trainable_modules = [
                'cls_adapter',
                'cls_head',
                'predict_head',
                'featurePyramid_net',
                'fusion_layers',
                'source_head',
                'auto_weight'
            ]

        for name, param in model.named_parameters():
            if any(target in name for target in trainable_modules):
                param.requires_grad = True

        # 3. 打印确认，确保万无一失
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]

        print(f"❄️ 已冻结参数层数: {len(frozen_params)} (包含 ConvNeXt, SegFormer, Source_head 等)")
        print(f"❄️ {frozen_params}")
        print(f"🔥 正在训练的参数层数: {len(trainable_params)} (包含 FPN, Fusion, Cls_head, Predict_head)")
        print(f"🔥 {trainable_params} ")
        # =================================================================


        # 旧版 暂时保留
        # =================================================================
        # 如有必要 冻结原有网络，只训练新加的 source_head 和 auto_weight
        # =================================================================
        # if args.freeze_backbone:
        #     print("❄️ [Policy Activation] Freezing the backbone network to protect the existing mask fake detection capabilities...")
        #     for name, param in model.named_parameters():
        #         # 只要参数名字里不包含 'source_head' 和 'auto_weight'，就停止它的梯度计算
        #         if 'source_head' not in name and 'auto_weight' not in name:
        #             param.requires_grad = False
        # else:
        #     print("🔥 [Policy Activation] Backbone network fully unfrozen")
        #     for name, param in model.named_parameters():
        #         param.requires_grad = True  # 确保所有参数都参与训练

        # =================================================================
        # 冻结策略 2：保护溯源能力，只微调其它模块
        # =================================================================
        # if args.freeze_source:
        #     print("🧊 [Policy Activation] Freezing source_head & auto_weight to preserve current source tracing ability...")
        #     for name, param in model.named_parameters():
        #         if 'source_head' in name or 'auto_weight' in name:
        #             param.requires_grad = False

        # 检查一下哪些参数在训练
        # trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        # if args.freeze_backbone or args.freeze_source:
        #     print(f"🔥 The only parameters involved in training are: {trainable_params}")
        # else:
        #     print(f"🔥 All parameters are involved in training, "
        #           f"resulting in a total of {len(trainable_params)} parameter layers.")
            # print(f"🔥 参与训练的参数共有: {trainable_params}")
        # =================================================================

        
    else:
        # from scratch
        ckpt_path = ''
        args.start_epoch = 0

        print("#"*80)
        print("Training from scratch")
    
        print(f"[IMPORTANT] resume = {args.resume}, ckpt_path = {ckpt_path}, start_epoch = {args.start_epoch}")

    if global_rank == 0:
        count_parameters(model)
    
    #exit(0)
    model.to(device)

    model_without_ddp = model
    #print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    args.opt='AdamW'
    args.betas=(0.9, 0.999)
    args.momentum=0.9
    optimizer  = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_evaluate_metric_value = 0

    
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            log_per_epoch_count=args.log_per_epoch_count,
            args=args,
            evaluator_list=evaluator_list
        )
        
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch)
            
        optimizer.zero_grad()
        if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
            values = {}
            for name, data_loader_test in test_data_loader.items():
                test_stats = test_one_epoch(
                    model, 
                    data_loader = data_loader_test, 
                    evaluator_list=evaluator_list,
                    device = device, 
                    epoch = epoch,
                    name = name, 
                    log_writer=log_writer,
                    args = args,
                    is_test = False,
                )
                one_metric_value = {}
                for evaluate_metric_for_ckpt in evaluator_list:
                    evaluate_metric_for_ckpt_name = evaluate_metric_for_ckpt.name
                    evaluate_metric_value = test_stats[evaluate_metric_for_ckpt_name]
                    one_metric_value[evaluate_metric_for_ckpt_name] = evaluate_metric_value
                values[name] = one_metric_value

            metrics_dict = {metric: {dataset: values[dataset][metric] for dataset in values} for metric in {m for d in values.values() for m in d}}
            metric_means = {metric: np.mean(list(datasets.values())) for metric, datasets in metrics_dict.items()}

            evaluate_metric_value = np.mean(list(metric_means.values()))

            if evaluate_metric_value > best_evaluate_metric_value :
                best_evaluate_metric_value = evaluate_metric_value
                print(f"Best {' '.join([evaluator.name for evaluator in evaluator_list])} = {best_evaluate_metric_value}")
                if epoch > 0:
                    misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            else:
                print(f"Average {' '.join([evaluator.name for evaluator in evaluator_list])} = {evaluate_metric_value}")
            if log_writer is not None:
                for metric, datasets in metrics_dict.items():
                    log_writer.add_scalars(f'{metric}_Metric', datasets, epoch)
                log_writer.add_scalar('Average', evaluate_metric_value, epoch)
            log_stats =  {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        torch.cuda.empty_cache()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

class Logger(object):
    def __init__(self, filename='console_log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # 强制刷新屏幕
        self.log.write(message)
        self.log.flush()       # 强制写入文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == '__main__':


    import torch.distributed as dist
    from datetime import timedelta
    # 防止出现uninitialized value报错
    # 反正我现在只有1张卡

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    if not dist.is_initialized():
        store = dist.TCPStore("127.0.0.1", 12345, 1, is_master=True, timeout=timedelta(seconds=30), use_libuv=False)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)

    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
