import argparse
import datetime
import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode

import nanobrain.misc as misc
from nanobrain.model import ViTMAE3D, patches_to_volume
from nanobrain.transforms import brle_to_dense, normalize_values
from nanobrain.visualization import plot_mask_pred

logger = logging.getLogger()

DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"


def main(args: DictConfig):
    # setup
    assert int(os.environ.get("WORLD_SIZE", 1)) == 1, "distributed training not supported"
    device = torch.device(args.device)
    misc.random_seed(args.seed)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    misc.setup_logging(logger, output_dir / "log.txt")

    logger.info("pretraining nanobrain")
    logger.info(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"cwd: {Path.cwd()}")
    logger.info(f"sha: {misc.git_sha()}")
    logger.info(f"config:\n{OmegaConf.to_yaml(args)}")

    # data loader
    dataset = Fomo300kDataset(args.data_root, args.train_filelist)
    logger.info(f"train dataset: {len(dataset)} images")
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list,  # no collate, variable shapes. lambda can't be pickled to workers
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # model
    model = ViTMAE3D.from_config(args)
    model.to(device)
    logger.info(f"model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"num params: {num_params / 1e6:.1f}M")

    batch_flops = count_batch_flops(args, model, device)
    logger.info(f"flops per batch (fwd + bwd): {batch_flops / 1e12:.2f} TFLOP")

    if args.compile:
        # in place compile keeps the state dict keys
        model.compile()

    # optimizer
    # batch size counts sequences, following mae_st which scales by repeated samples
    total_batch_size = args.batch_size * args.num_samples * args.accum_iter
    logger.info(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} images x {args.num_samples} samples x {args.accum_iter} accum"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        logger.info(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        logger.info(f"lr: {args.lr:.2e}")

    param_groups = misc.get_param_groups(model)
    misc.update_lr(param_groups, args.lr)
    misc.update_wd(param_groups, args.weight_decay)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    optimizer = torch.optim.AdamW(param_groups, betas=betas, fused=args.fused_adamw)

    epoch_num_batches = len(train_loader)
    steps_per_epoch = math.ceil(epoch_num_batches / args.accum_iter)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    lr_schedule = misc.WarmupThenCosine(
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
    )
    logger.info(f"full schedule: epochs = {args.epochs} (steps = {total_steps})")
    logger.info(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None

    # load checkpoint/resume training
    misc.load_model(args, model, optimizer, loss_scaler)

    logger.info(f"start training for {args.epochs} epochs")
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            loss_scaler,
            lr_schedule,
            epoch,
            device,
            batch_flops,
        )

        merged_stats = {"epoch": epoch, **train_stats}
        with (output_dir / "log.json").open("a") as f:
            print(json.dumps(merged_stats), file=f)

        misc.save_model(args, epoch, model, optimizer, loss_scaler)

    total_time = time.monotonic() - start_time
    logger.info(f"done! training time: {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
    batch_flops: float,
):
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", misc.SmoothedValue())
    metric_logger.add_meter("loss", misc.SmoothedValue())
    metric_logger.add_meter(
        "vol/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "tflop/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "watts", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    # cpu time, the step has no syncs. prep is host to device copy and gpu preprocess
    metric_logger.add_meter("prep", misc.SmoothedValue(fmt="{avg:.4f}"))
    metric_logger.add_meter("model", misc.SmoothedValue(fmt="{avg:.4f}"))
    header = f"Train: [{epoch}]"

    throughput = misc.ThroughputMeter(
        {"vol": args.batch_size, "tflop": batch_flops / 1e12}, device=device
    )

    epoch_num_batches = len(data_loader)
    steps_per_epoch = math.ceil(epoch_num_batches / args.accum_iter)

    print_freq = args.get("print_freq", 100) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10
    amp_dtype = getattr(torch, args.amp_dtype)
    use_cuda = device.type == "cuda"

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        batch_step = batch_idx + 1
        log_step = batch_step % print_freq == 0 or batch_step == num_batches
        update_in_epoch = batch_idx // args.accum_iter
        group_size = min(args.accum_iter, num_batches - update_in_epoch * args.accum_iter)
        need_update = batch_step % args.accum_iter == 0 or batch_step == num_batches
        global_step = epoch * steps_per_epoch + update_in_epoch
        lr = lr_schedule[global_step]
        if need_update:
            misc.update_lr(optimizer.param_groups, lr)

        prep_start = time.perf_counter()
        if use_cuda:
            batch = misc.send_data(batch, device)

        # run volume processing on gpu, not cpu workers
        samples = [model.preprocess(decode_sample(sample)) for sample in batch]
        images = torch.stack([sample["image"] for sample in samples])
        masks = torch.stack([sample["mask"] for sample in samples])
        model_start = time.perf_counter()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=args.num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
            )

        loss_for_log = loss.detach()
        torch._assert_async(torch.isfinite(loss_for_log), "non-finite loss")

        grad_norm = misc.backward_step(
            loss / group_size,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        model_end = time.perf_counter()

        # log tensors without syncing, values are only read on log steps
        metric_logger.update(loss=loss_for_log)
        metric_logger.update(prep=model_start - prep_start, model=model_end - model_start)
        if need_update:
            metric_logger.update(lr=lr, grad=grad_norm)

        throughput.step()
        if log_step:
            metric_logger.update(**throughput.compute())

        if need_update and log_step and args.wandb:
            wandb.log(
                {
                    "train/loss": metric_logger.loss.value,
                    "train/lr": lr,
                    "train/grad": metric_logger.grad.value,
                    "train/vol_per_sec": metric_logger.meters["vol/s"].value,
                    "train/tflop_per_sec": metric_logger.meters["tflop/s"].value,
                    "train/watts": metric_logger.meters["watts"].value,
                },
                step=int(1000 * (epoch + batch_step / epoch_num_batches)),
            )

    # plot a few images from the last batch
    num_examples = args.plot_nrow * args.plot_ncol
    is_plot_epoch = args.plot_period and (epoch % args.plot_period == 0 or epoch == args.epochs - 1)
    if num_examples and is_plot_epoch:
        model.eval()
        example_images = images[:num_examples]
        example_masks = masks[:num_examples]
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp),
        ):
            _, state = model(
                example_images,
                example_masks,
                num_visible=args.num_visible,
                num_predict=args.plot_num_predict,
                num_samples=1,
            )
        model.train()
        plot_paths = make_plots(args, example_images, example_masks, state, epoch)
        if args.wandb:
            wandb.log(
                {f"train/{name}": wandb.Image(str(path)) for name, path in plot_paths.items()},
                step=1000 * (epoch + 1),
            )

    logger.info(f"Averaged stats: {metric_logger}")
    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


class Fomo300kDataset(Dataset):
    def __init__(self, root: str | Path, filelist: str | Path):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().split()
        members = pd.read_parquet(
            self.root / "tables" / "shards.parquet", columns=["path", "shard", "offset", "size"]
        )
        members = members.set_index("path").loc[self.paths]
        self.shards = members["shard"].tolist()
        self.offsets = members["offset"].to_numpy()
        self.sizes = members["size"].to_numpy()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        # npz members are stored uncompressed, so read them straight from their offset
        with open(self.root / self.shards[index], "rb") as shard:
            shard.seek(self.offsets[index])
            data = shard.read(self.sizes[index])
        with np.load(io.BytesIO(data)) as npz:
            sample = {
                "path": self.paths[index],
                "values": torch.from_numpy(npz["values"]),
                "scale": float(npz["scale"]),
                "mask_rle": torch.from_numpy(npz["mask_rle"]),
                "shape": tuple(npz["shape"].tolist()),
                "affine": npz["affine"],
            }
        return sample


def decode_sample(sample: dict) -> dict:
    mask = brle_to_dense(sample["mask_rle"], shape=sample["shape"])
    # normalize the in-mask values before scattering, normalizing the dense image would sync.
    # images are already cropped to the mask. eval does the same in ViTMAE3D.transform
    values = normalize_values(sample["values"].float() * sample["scale"])
    image = torch.zeros(mask.shape, dtype=values.dtype, device=values.device)
    image.masked_scatter_(mask, values)
    return {"image": image, "mask": mask, "affine": sample["affine"]}


def count_batch_flops(args: DictConfig, model: nn.Module, device: torch.device) -> float:
    # fwd + bwd flops of one training batch, on dummy images where every patch is in the mask
    amp_dtype = getattr(torch, args.amp_dtype)
    images = torch.zeros(args.batch_size, *args.grid_size, device=device)
    masks = torch.ones(args.batch_size, *args.grid_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=args.num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
            )
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def make_plots(
    args: DictConfig,
    images: Tensor,
    masks: Tensor,
    state: dict[str, Tensor],
    epoch: int,
) -> dict[str, Path]:
    grid_size = tuple(args.grid_size)
    patch_size = args.patch_size
    vis_ids = state["vis_ids"]
    target_ids = state["target_ids"]
    pred = patches_to_volume(state["pred_patches"].float(), target_ids, grid_size, patch_size)
    visible_mask = patches_to_volume(
        torch.ones_like(state["vis_patches"]), vis_ids, grid_size, patch_size
    )
    pred_mask = patches_to_volume(
        torch.ones_like(state["target_patches"]), target_ids, grid_size, patch_size
    )

    images = images.float().cpu().numpy()
    masks = masks.cpu().numpy()
    pred = pred.cpu().numpy()
    visible_mask = visible_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    plot_dir = Path(args.output_dir) / "plots"
    plot_paths = {}
    for view in ["sag", "cor", "ax"]:
        fig = plot_mask_pred(
            images,
            pred,
            visible_mask,
            pred_mask=pred_mask,
            img_mask=masks,
            view=view,
            nrow=args.plot_nrow,
            ncol=args.plot_ncol,
        )
        path = plot_dir / f"mae_{view}_{epoch:05d}.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[f"mae_{view}"] = path
    return plot_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
