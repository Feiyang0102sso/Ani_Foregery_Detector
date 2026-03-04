import math
import sys
from typing import Iterable

import torch

import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.training_scripts.schedular.cos_lr_schedular import adjust_learning_rate  # TODO

from IMDLBenCo.datasets import denormalize
from contextlib import nullcontext


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    log_per_epoch_count=20,
                    args=None,
                    evaluator_list=None):  # <--- [新增] 1. 接收 evaluator_list 参数

    model.train(True)

    # ++++++ 【清理硬编码 - 引入按需冻结机制】 ++++++
    # 根据模块是否参与训练 (requires_grad) 动态设置 train() 还是 eval() 模式
    # 这样被冻结的骨干（包含 BN 层）就不会在野图训练中跑偏参数
    module_to_freeze = model.module if hasattr(model, 'module') else model
    for name, module in module_to_freeze.named_modules():
        # 获取该模块下所有的参数
        params = list(module.parameters(recurse=False))
        # 只要该模块直接包含的任何参数需要求导，就设为 train()，否则强制 eval()
        if any(p.requires_grad for p in params):
            module.train()
        elif len(params) > 0:
            module.eval()
            
    # 特别地，显式把不参加训练的整体骨干也强制 eval 避免意外的 drop_path 等行为
    if not any(p.requires_grad for p in module_to_freeze.convnext.parameters()):
        module_to_freeze.convnext.eval()
    if not any(p.requires_grad for p in module_to_freeze.segformer.parameters()):
        module_to_freeze.segformer.eval()
    # ++++++ 【机制更新完毕】 ++++++

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # <--- [新增] 2. 每个 epoch 开始前，清空评估器的历史缓存
    if evaluator_list is not None:
        for evaluator in evaluator_list:
            evaluator.recovery()

    amp_placeholder = torch.cuda.amp.autocast() if args.if_not_amp else nullcontext()

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    total_step = len(data_loader)
    log_period = total_step / log_per_epoch_count
    # Start training
    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # move to device
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        torch.cuda.synchronize()

        with amp_placeholder:
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            output_dict = model(**data_dict,
                                if_predcit_label=args.if_predict_label,
                                disable_source_loss = args.disable_source_loss
                                )
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            loss = output_dict['backward_loss']
            mask_pred = output_dict['pred_mask']

            # <--- [新增] 3. 将当批次的预测结果输入评估器 (必须加上 no_grad 避免梯度爆炸)
            if evaluator_list is not None:
                with torch.no_grad():
                    for evaluator in evaluator_list:
                        res = None
                        if "pixel" in evaluator.name.lower():
                            if 'mask' in data_dict and 'pred_mask' in output_dict:
                                res = evaluator.batch_update(predict=output_dict['pred_mask'], mask=data_dict['mask'])
                        elif "image" in evaluator.name.lower():
                            if 'label' in data_dict and 'pred_label' in output_dict:
                                res = evaluator.batch_update(predict_label=output_dict['pred_label'],
                                                       label=data_dict['label'])

                        #[新增] 处理溯源分类指标
                        elif "source" in evaluator.name.lower():
                            if 'source_label' in data_dict and 'pred_source' in output_dict:
                                res = evaluator.batch_update(predict_label=output_dict['pred_source'],
                                                        label=data_dict['source_label'])
                        
                        if res is not None:
                             if isinstance(res, torch.Tensor):
                                 if res.numel() > 1:
                                     val = res.mean().item()
                                 else:
                                     val = res.item()
                             else:
                                 val = res
                             metric_logger.update(**{evaluator.name: val})

            visual_loss = output_dict['visual_loss']
            visual_loss_item = {}
            for k, v in visual_loss.items():
                visual_loss_item[k] = v.item()

            visual_image = output_dict['visual_image']

            predict_loss = loss / accum_iter
        loss_scaler(predict_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        # save to log.txt
        metric_logger.update(lr=lr)

        metric_logger.update(**visual_loss_item)
        # metric_logger.update(predict_loss= predict_loss_value)
        # metric_logger.update(edge_loss= edge_loss_value)

        visual_loss_reduced = {}
        for k, v in visual_loss_item.items():
            visual_loss_reduced[k] = misc.all_reduce_mean(v)

        if log_writer is not None and (data_iter_step + 1) % max(int(log_period), 1) == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            # Tensorboard logging
            log_writer.add_scalar('lr', lr, epoch_1000x)

            for k, v in visual_loss_reduced.items():
                log_writer.add_scalar(f"train_loss/{k}", v, epoch_1000x)
            # log_writer.add_scalar('train_loss/predict_loss', loss_predict_reduce, epoch_1000x)
            # log_writer.add_scalar('train_loss/edge_loss', edge_loss_reduce, epoch_1000x)

    samples = data_dict['image']
    mask = data_dict['mask']

    if log_writer is not None:
        log_writer.add_images('train/image', denormalize(samples), epoch)
        log_writer.add_images('train/predict', mask_pred, epoch)
        log_writer.add_images('train/predict_thresh_0.5', (mask_pred > 0.5) * 1.0, epoch)
        log_writer.add_images('train/gt_mask', mask, epoch)

        for k, v in visual_image.items():
            log_writer.add_images(f'train/{k}', v, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 提取基础 stats
    res_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # <--- [新增] 4. Epoch 结束时，计算最终指标、打印、写入 Tensorboard 并合并到返回字典中
    if evaluator_list is not None:
        metrics_to_print = {}
        
        # Collect metrics from metric_logger (for pixel-level metrics)
        for evaluator in evaluator_list:
            if evaluator.name in metric_logger.meters:
                metrics_to_print[evaluator.name] = metric_logger.meters[evaluator.name].global_avg
        
        # Collect metrics from epoch_update (for image-level metrics)
        for evaluator in evaluator_list:
            metric_value = evaluator.epoch_update()
            if metric_value is not None:
                # 兼容 tensor 输出和普通数值输出
                val = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
                metrics_to_print[evaluator.name] = val

                # 写入到返回的字典里（后续会被保存到 log.txt 中）
                res_dict[f"{evaluator.name}"] = val

                # 写入 TensorBoard
                if log_writer is not None:
                    log_writer.add_scalar(f"Train_Metric/{evaluator.name}", val, epoch)
        
        misc.print_metrics_table(metrics_to_print, title=f"Train Results - Epoch {epoch}")

    return res_dict

# seems useless in this project
def train_one_epoch_imlvit4exp(model: torch.nn.Module,
                               data_loader: Iterable,
                               optimizer: torch.optim.Optimizer,
                               device: torch.device,
                               epoch: int,
                               loss_scaler,
                               log_writer=None,
                               log_per_epoch_count=20,
                               args=None,
                               evaluator_list=None):  # <--- [新增] 1. 接收 evaluator_list 参数
    model.train(True)

    # # ++++++ 【新增代码】 ++++++
    # # 哪怕整体是 train 模式，也要把冻结的骨干强制按回 eval 模式！
    # # 这样 Dropout 会失效，BatchNorm 不再更新 running_mean 和 running_var
    # module_to_freeze = model.module if hasattr(model, 'module') else model
    #
    # module_to_freeze.convnext.eval()
    # module_to_freeze.segformer.eval()
    # module_to_freeze.fusion_layers.eval()
    # module_to_freeze.featurePyramid_net.eval()
    # module_to_freeze.predict_head.eval()
    # module_to_freeze.cls_head.eval()
    # # 只保留 source_head 为 train 模式 (如果你要训练它的话)
    # module_to_freeze.source_head.train()
    # # ++++++ 【新增代码结束】 ++++++

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # <--- [新增] 2. 每个 epoch 开始前，清空评估器的历史缓存
    if evaluator_list is not None:
        for evaluator in evaluator_list:
            evaluator.recovery()

    amp_placeholder = torch.cuda.amp.autocast() if args.if_not_amp else nullcontext()

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    total_step = len(data_loader)
    log_period = total_step / log_per_epoch_count
    # Start training
    for data_iter_step, data_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # move to device
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        torch.cuda.synchronize()

        with amp_placeholder:
            output_dict = model(**data_dict,
                                if_predcit_label=args.if_predict_label
                                )

            loss = output_dict['backward_loss']
            mask_pred = output_dict['pred_mask']

            # <--- [新增] 3. 将当批次的预测结果输入评估器
            if evaluator_list is not None:
                with torch.no_grad():
                    for evaluator in evaluator_list:
                        res = None
                        if "pixel" in evaluator.name.lower():
                            if 'mask' in data_dict and 'pred_mask' in output_dict:
                                res = evaluator.batch_update(predict=output_dict['pred_mask'], mask=data_dict['mask'])
                        elif "image" in evaluator.name.lower():
                            if 'label' in data_dict and 'pred_label' in output_dict:
                                res = evaluator.batch_update(predict_label=output_dict['pred_label'],
                                                       label=data_dict['label'])

                        #[新增] 处理溯源分类指标
                        elif "source" in evaluator.name.lower():
                            if 'source_label' in data_dict and 'pred_source' in output_dict:
                                res = evaluator.batch_update(predict_label=output_dict['pred_source'],
                                                        label=data_dict['source_label'])
                        
                        if res is not None:
                             if isinstance(res, torch.Tensor):
                                 if res.numel() > 1:
                                     val = res.mean().item()
                                 else:
                                     val = res.item()
                             else:
                                 val = res
                             metric_logger.update(**{evaluator.name: val})

            visual_loss = output_dict['visual_loss']
            visual_loss_item = {}
            for k, v in visual_loss.items():
                visual_loss_item[k] = v.item()

            visual_image = output_dict['visual_image']

            predict_loss = loss / accum_iter
        loss_scaler(predict_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        # save to log.txt
        metric_logger.update(lr=lr)

        metric_logger.update(**visual_loss_item)
        # metric_logger.update(predict_loss= predict_loss_value)
        # metric_logger.update(edge_loss= edge_loss_value)

        visual_loss_reduced = {}
        for k, v in visual_loss_item.items():
            visual_loss_reduced[k] = misc.all_reduce_mean(v)

        if log_writer is not None and (data_iter_step + 1) % max(int(log_period), 1) == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            # Tensorboard logging
            log_writer.add_scalar('lr', lr, epoch_1000x)

            for k, v in visual_loss_reduced.items():
                log_writer.add_scalar(f"train_loss/{k}", v, epoch_1000x)
            # log_writer.add_scalar('train_loss/predict_loss', loss_predict_reduce, epoch_1000x)
            # log_writer.add_scalar('train_loss/edge_loss', edge_loss_reduce, epoch_1000x)

        if data_iter_step % 100 == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            print(f"weight_low_freq={model.module.weight_low_freq}, weight_high_freq={model.module.weight_high_freq}")
            if log_writer is not None:
                weight_low_freq = model.module.weight_low_freq.item()
                weight_high_freq = model.module.weight_high_freq.item()
                log_writer.add_scalar('weight_low_freq', weight_low_freq, epoch_1000x)
                log_writer.add_scalar('weight_high_freq', weight_high_freq, epoch_1000x)

    samples = data_dict['image']
    mask = data_dict['mask']

    if log_writer is not None:
        log_writer.add_images('train/image', denormalize(samples), epoch)
        log_writer.add_images('train/predict', mask_pred, epoch)
        log_writer.add_images('train/predict_thresh_0.5', (mask_pred > 0.5) * 1.0, epoch)
        log_writer.add_images('train/gt_mask', mask, epoch)

        for k, v in visual_image.items():
            log_writer.add_images(f'train/{k}', v, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 提取基础 stats
    res_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # <--- [新增] 4. Epoch 结束时，计算最终指标、打印、写入 Tensorboard 并合并到返回字典中
    if evaluator_list is not None:
        metrics_to_print = {}
        
        # Collect metrics from metric_logger (for pixel-level metrics)
        for evaluator in evaluator_list:
            if evaluator.name in metric_logger.meters:
                metrics_to_print[evaluator.name] = metric_logger.meters[evaluator.name].global_avg
        
        # Collect metrics from epoch_update (for image-level metrics)
        for evaluator in evaluator_list:
            metric_value = evaluator.epoch_update()
            if metric_value is not None:
                # 兼容 tensor 输出和普通数值输出
                val = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
                metrics_to_print[evaluator.name] = val

                # 写入到返回的字典里（后续会被保存到 log.txt 中）
                res_dict[f"{evaluator.name}"] = val

                # 写入 TensorBoard
                if log_writer is not None:
                    log_writer.add_scalar(f"Train_Metric/{evaluator.name}", val, epoch)
        
        misc.print_metrics_table(metrics_to_print, title=f"Train Results - Epoch {epoch}")

    return res_dict