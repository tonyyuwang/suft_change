import sys

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"  # on

# 加了一行文字 test2

# 添加了注释

# 测试新的

from pathlib import Path

import torch.cuda
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.distributed

# from torch.utils.tensorboard import SummaryWriter

import utils.torch as ptu
from data.factory import create_dataset
from utils import distributed
from model.utils import num_params
from optim.factory import create_optimizer, create_scheduler

from model.vision_transformer import SwinUnet
from engine import train_one_epoch

import config

from timm.utils import NativeScaler
from contextlib import suppress
import os

import collections


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory", default="logs/without_correct")
@click.option("--dataset", default='coco', type=str)
@click.option('--dataset_dir', default='/data/wy/ImageNet',type=str)
@click.option("--im-size", default=256, type=int, help="dataset resize size")
@click.option("--crop-size", default=256, type=int)
@click.option("--window-size", default=256, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_tiny_patch16_384", type=str)       # try this, and freeze first several blocks.
@click.option("--decoder", default="mask_transformer", type=str)
@click.option("--optimizer", default="adam", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)


# @click.option("--batch-size", default=None, type=int)
@click.option("--batch-size", default=None, type=int)


@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)

@click.option("--resume/--no-resume", default=True, is_flag=True)


@click.option('--local_rank', type=int, default=3)


@click.option('--only_test', type=bool, default=False)
@click.option('--add_mask', type=bool, default=True)        # valid
@click.option('--partial_finetune', type=bool, default=False)       # compare validation, last finetune all blocks.

@click.option('--add_l1_loss', type=bool, default=True)            # add after classification.

@click.option('--l1_weight', type=float, default=10)
@click.option('--color_position', type=bool, default=True)     # add color position in color token.
@click.option('--change_mask', type=bool, default=False)        # change mask, omit the attention between color tokens.
@click.option('--color_as_condition', type=bool, default=False)     # use self-attn to embedding color tokens, and use color to represent patch tokens.
@click.option('--multi_scaled', type=bool, default=False)       # multi-scaled decoder.
@click.option('--downchannel', type=bool, default=False)        # multi-scaled, upsample+downchannel. (should be correct??)
@click.option('--add_conv', type=bool, default=True)       # add conv after transformer blocks.
@click.option('--before_classify', type=bool, default=False)        # classification at 16x16 resolution, and use CNN upsampler for 256x256, then use l1-loss.
@click.option('--l1_conv', type=bool, default=True)                # patch--upsample--> [B, 256x256, 180]--conv3x3-> [B, 256x256, 2]
@click.option('--l1_linear', type=bool, default=False)          # pacth: [B, 16x16, 180]---linear-> [B, 16x16, 2x16x16]
@click.option('--add_fm', type=bool, default=False)             # add Feature matching loss.
@click.option('--fm_weight', type=float, default=1)
@click.option('--add_edge', type=bool, default=False)       # add sobel-conv to extract edge.
@click.option('--edge_loss_weight', type=float, default=0.05)     # edge_loss_weight.
@click.option('--mask_l_num', type=int, default=4)          # mask for L ranges: 4, 10, 20, 50, 100
@click.option('--n_blocks', type=int, default=1)        # per block have 2 layers. block_num = 2
@click.option('--n_layers', type=int, default=2)
@click.option('--without_colorattn', type=bool, default=False)
@click.option('--without_colorquery', type=bool, default=False)
@click.option('--without_classification', type=bool, default=False)
@click.option('--color_token_num', type=int, default=313)
@click.option('--sin_color_pos', type=bool, default=False)

@click.option('--with_regression', type=bool, default=False)
@click.option('--freeze', type=bool, default=False)

def main(
    log_dir,
    dataset,
    dataset_dir,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    decoder,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    drop_path,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    amp,
    resume,
    freeze,
    local_rank,
    only_test,
    add_mask,
    partial_finetune,
    add_l1_loss,
    l1_weight,
    color_position,
    change_mask,
    color_as_condition,
    multi_scaled,
    downchannel,
    add_conv,
    before_classify,
    l1_conv,
    l1_linear,
    add_fm,
    fm_weight,
    add_edge,
    edge_loss_weight,
    mask_l_num,
    n_blocks,
    n_layers,
    without_colorattn,
    without_colorquery,
    without_classification,
    color_token_num,
    sin_color_pos,
    with_regression,
):
    # start distributed mode
    # if torch.cuda.is_available():
    #     print(torch.cuda.device_count())
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'



    local_rank = int(os.environ["LOCAL_RANK"])  # on
    ptu.set_gpu_mode(True, local_rank)  # on
    torch.distributed.init_process_group(backend="nccl", init_method='env://')  # on
    print("cuda_device_count:", torch.cuda.device_count())

    # if torch.cuda.is_available():
    #     device_count = torch.cuda.device_count()
    #     print("cuda_device_count:", torch.cuda.device_count())

    # ptu.set_gpu_mode(True, local_rank)

    # tb_writer = SummaryWriter('/logs')


    # distributed.init_process()
    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]        # 256
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)       # 256
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)
    if not dataset_dir:
        dataset_dir = dataset_cfg.get('dataset_dir', None)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout          # 0
    model_cfg["drop_path_rate"] = drop_path     # 0.1
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]

    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    print('ptu.world_size', ptu.world_size)
    batch_size = world_batch_size // ptu.world_size
    print('bs', batch_size)
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=6,  # default = 10
            dataset_dir=dataset_dir,
            add_mask=add_mask,
            patch_size=model_cfg["patch_size"],
            change_mask=change_mask,
            multi_scaled=multi_scaled,
            mask_num=mask_l_num,
            n_cls=color_token_num,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = log_dir / 'third_checkpoint_epoch_1_iter_205000.pth'  # tiny.

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = color_token_num
    net_kwargs['partial_finetune'] = partial_finetune
    net_kwargs['decoder']['add_l1_loss'] = add_l1_loss
    net_kwargs['decoder']['color_position'] = color_position
    net_kwargs['decoder']['change_mask'] = change_mask
    net_kwargs['decoder']['color_as_condition'] = color_as_condition
    net_kwargs['decoder']['multi_scaled'] = multi_scaled
    net_kwargs['decoder']['crop_size'] = crop_size
    net_kwargs['decoder']['downchannel'] = downchannel
    net_kwargs['decoder']['add_conv'] = add_conv
    net_kwargs['decoder']['before_classify'] = before_classify
    net_kwargs['decoder']['l1_conv'] = l1_conv
    net_kwargs['decoder']['l1_linear'] = l1_linear
    net_kwargs['decoder']['add_edge'] = add_edge
    net_kwargs['decoder']['n_blocks'] = n_blocks
    net_kwargs['decoder']['n_layers'] = n_layers
    net_kwargs['decoder']['without_colorattn'] = without_colorattn
    net_kwargs['decoder']['without_colorquery'] = without_colorquery
    net_kwargs['decoder']['without_classification'] = without_classification
    net_kwargs['decoder']['sin_color_pos'] = sin_color_pos

    model = SwinUnet(config=cfg, with_regression=with_regression)
    model.load_from(cfg)
    model.to(ptu.device)
    if ptu.distributed:
        print('Distributed:', ptu.distributed)
        # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)  # on

        # optimizer
        optimizer_kwargs = variant["optimizer_kwargs"]
        optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
        optimizer_kwargs["iter_warmup"] = 0.0
        opt_args = argparse.Namespace()
        opt_vars = vars(opt_args)
        for k, v in optimizer_kwargs.items():
            opt_vars[k] = v
        optimizer = create_optimizer(opt_args, model, partial_finetune)
        lr_scheduler = create_scheduler(opt_args, optimizer)
        num_iterations = 0
        amp_autocast = suppress
        loss_scaler = None
        #  autocast + gradscaler
        if amp:
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()

        # resume
        if resume and checkpoint_path.exists():
            print("resuming:", True)
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # 载入固定参数
            pretrained_dict = checkpoint['model']  # ??
            model_dict = model.state_dict()

            # 有问题
            # state_dict = {'attention_map.' + k: v for k, v in pretrained_dict.items() if
            #               'attention_map.' + k in model_dict.keys()}
            # state_dict = {'module.' + k: v for k, v in pretrained_dict.items() if 'module.' + k in model_dict.keys()}
            state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}  # on

            # print("state_dict", state_dict)  # 空的？？？
            # #
            # for name in state_dict.keys():
            #     print("state_dict.name", name)
            # for name in pretrained_dict.keys():
            #     print("pretrained_dict.name", name)
            # for name in model_dict.keys():
            #     print("model_dict.name", name)

            model_dict.update(state_dict)
            model.load_state_dict(model_dict)

            # dict_name = list(model_dict)
            # for i, p in enumerate(dict_name):
            #     print(i, p)

            # 冻结权重
            if freeze:
                print("freeze:", True)
                for p in model.swin_unet.parameters():
                    p.requires_grad = False
                # for p in model.module.decoder.parameters():
                #     p.requires_grad = False

            # model.load_state_dict(checkpoint["model"])    # for pos encoding(default = on)
            # optimizer.load_state_dict(checkpoint["optimizer"])  # on
            if loss_scaler and "loss_scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["loss_scaler"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1

        # for k, v in model.swin_unet.named_parameters():  # 查看是否冻结成功
        #     print('{}: {}'.format(k, v.requires_grad))
        # for k, v in model.correct.named_parameters():
        #     print('{}: {}'.format(k, v.requires_grad))

        # save config
        variant_str = yaml.dump(variant)
        # print(f"Configuration:\n{variant_str}")
        variant["net_kwargs"] = net_kwargs
        variant["dataset_kwargs"] = dataset_kwargs
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "variant.yml", "w") as f:
            f.write(variant_str)

        # train
        start_epoch = variant["algorithm_kwargs"]["start_epoch"]
        num_epochs = variant["algorithm_kwargs"]["num_epochs"]
        # eval_freq = variant["algorithm_kwargs"]["eval_freq"]

        model_without_ddp = model
        if hasattr(model, "module"):
            model_without_ddp = model.module

        # val_seg_gt = val_loader.dataset.get_gt_seg_maps()

        print(f"Train dataset length: {len(train_loader.dataset)}")
        # print(f"Val dataset length: {len(val_loader.dataset)}")
        # print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
        # print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")
        print(f"swinunet parameters: {num_params(model)}")

        # print(f"Regression parameters: {num_params(model_without_ddp.regression)}")

        for epoch in range(20):
            torch.cuda.empty_cache()
            # train for one epoch
            print('Training...', epoch)
            train_logger, loss_epoch, loss_l1_epoch, lr_epoch = train_one_epoch(
                model,
                train_loader,
                optimizer,
                lr_scheduler,
                epoch,
                amp_autocast,
                loss_scaler,
                add_mask,
                add_l1_loss,
                l1_weight,
                partial_finetune,
                l1_conv,
                l1_linear,
                add_edge,
                edge_loss_weight,
                without_classification,
                log_dir,
                with_regression,
            )

            tags = ["train_loss", "train_l1_loss", "learning_rate"]
            # tb_writer.add_scalar(tags[0], loss_epoch, epoch)
            # tb_writer.add_scalar(tags[1], loss_l1_epoch, epoch)
            # tb_writer.add_scalar(tags[2], lr_epoch, epoch)

            # # # save checkpoint
            if ptu.dist_rank == 0:
                snapshot = dict(
                    model=model_without_ddp.state_dict(),
                    optimizer=optimizer.state_dict(),
                    n_cls=313,
                    lr_scheduler=lr_scheduler.state_dict(),
                )
                if loss_scaler is not None:
                    snapshot["loss_scaler"] = loss_scaler.state_dict()
                snapshot["epoch"] = epoch
                save_path = os.path.join(log_dir, 'first_checkpoint_epoch_%d.pth' % (epoch))
                torch.save(snapshot, save_path)
                print('save model into:', save_path)

            # log stats
            if ptu.dist_rank == 0:
                train_stats = {
                    k: meter.global_avg for k, meter in train_logger.meters.items()
                }

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    "epoch": epoch,
                    "num_updates": (epoch + 1) * len(train_loader),
                }

                with open(log_dir / "with_regression_log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        distributed.barrier()
        distributed.destroy_process()
        # sys.exit(1)

if __name__ == '__main__':
    main()
