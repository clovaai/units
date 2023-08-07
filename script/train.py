import warnings

warnings.filterwarnings("ignore")

import torch
from tensorfn import distributed as dist
from tensorfn import get_logger, load_arg_config
from tensorfn.config import instantiate
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader

from units.config import E2EConfig
from units.dataset import MultitaskCollator, MultitaskDataset, WeightedDataset
from units.train_fn import add_weight_decay, sample_data, wd_skip_fn
from units.transform import Compose


def make_dataset(datasources, weighted=True):
    if not isinstance(datasources, (list, tuple)):
        datasources = [datasources]

    sources = []
    ratios = []
    names = []

    for datasource in datasources:
        source_fn = instantiate(datasource.datasource)

        for name in datasource.sources:
            if isinstance(name, str):
                ratio = 1

            else:
                name, ratio = name

            sources.append(source_fn(datasource.path, name))
            ratios.append(ratio)
            names.append(name)

    if weighted:
        source = WeightedDataset(sources, ratios, names=names)

        if dist.is_primary():
            source.summary()

    else:
        source = ConcatDataset(sources)

        if dist.is_primary():
            for i, (name, s) in enumerate(zip(names, sources)):
                print(f"#{i} {name} total: {len(s)} ")

    return source


@torch.no_grad()
def evaluate(conf, mappers, loader, model, device, logger):
    is_train = model.training
    model.eval()

    # metrics = Metrics(
    #     conf.evaluate.eval_metrics, mappers, device, conf.evaluate.eval_metrics_option
    # )
    total = len(loader)

    for i, batch in enumerate(loader):
        out = model(batch.to(device))
        # metrics(batch, out)

        if dist.is_primary() and i % conf.log_freq == 0:
            logger.info(f"evaluation [{i}/{total}]")

    # res = metrics.compute()
    report = {}

    if is_train:
        model.train()

    return report


def main(conf):
    device = "cuda"
    conf.distributed = conf.n_gpu > 1

    logger = get_logger(mode=conf.logger)
    logger.info(conf.dict())

    model = instantiate(conf.model).to(device)
    model.train()

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
            find_unused_parameters=True,
        )

    source = make_dataset(conf.training.datasources)
    mappers = instantiate(conf.training.mappers)
    train_set = MultitaskDataset(
        source, mappers, transform=Compose(instantiate(conf.training.transform))
    )
    train_collator = MultitaskCollator(mappers)
    batch_size = conf.training.loader.batch_size // dist.get_world_size()

    train_loader = DataLoader(
        train_set,
        batch_size,
        num_workers=conf.training.loader.num_workers,
        sampler=dist.data_sampler(
            train_set, shuffle=True, distributed=conf.distributed
        ),
        collate_fn=train_collator,
    )

    source = make_dataset(conf.evaluate.datasources, weighted=False)
    valid_set = MultitaskDataset(
        source, mappers, transform=Compose(instantiate(conf.evaluate.transform))
    )
    valid_collator = MultitaskCollator(mappers, evaluate=True)
    val_batch_size = conf.training.val_loader.batch_size // dist.get_world_size()

    valid_loader = DataLoader(
        valid_set,
        val_batch_size,
        num_workers=conf.training.val_loader.num_workers,
        sampler=dist.data_sampler(
            valid_set, shuffle=False, distributed=conf.distributed
        ),
        collate_fn=valid_collator,
    )

    parameters, _ = add_weight_decay(
        model.named_parameters(),
        conf.training.weight_decay,
        wd_skip_fn(conf.training.wd_skip_fn),
    )
    optimizer = instantiate(conf.training.optimizer, parameters)
    scheduler = instantiate(conf.training.scheduler, optimizer)
    checker = instantiate(conf.checker)
    checker.catalog(conf)

    start_i = 0

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        if conf.finetune:
            logger.info(model.load_state_dict(ckpt["model"], strict=False))

        else:
            model.load_state_dict(ckpt["model"])
            start_i = ckpt["step"]

            if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                scheduler.load_state_dict(ckpt["scheduler"])

            if "optimizer" in ckpt and ckpt["optimizer"] is not None:
                optimizer.load_state_dict(ckpt["optimizer"])

    loader = sample_data(train_loader)
    grad_scaler = GradScaler(enabled=conf.training.mixed_precision)

    for i in range(start_i, conf.training.n_iter + 1):
        checker.checkpoint(
            {
                "model": model.state_dict(),
                "conf": conf.dict(),
                "optimizer": optimizer.state_dict()
                if hasattr(optimizer, "state_dict")
                else None,
                "scheduler": scheduler.state_dict()
                if hasattr(scheduler, "state_dict")
                else None,
                "step": i,
            },
            f"ckpt-{str(i).zfill(6)}.pt",
        )
        batch = next(loader).to(device)
        optimizer.zero_grad()

        with autocast(enabled=conf.training.mixed_precision):
            out = model(batch)

        grad_scaler.scale(out["total_loss"]).backward()
        grad_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), conf.training.clip_grad)
        scheduler.step()
        parameters[-1]["lr"] *= 0.1
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if dist.is_primary():
            if i % conf.log_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                losses = {"lr": lr}
                for k, v in out.items():
                    if not k.endswith("_loss"):
                        continue

                    losses[k.replace("_loss", "", 1)] = v.item()

                checker.log(**losses, step=i)

        if i % conf.evaluate.eval_freq == 0:
            checker.checkpoint(
                {
                    "model": model.state_dict(),
                    "conf": conf.dict(),
                    "optimizer": optimizer.state_dict()
                    if hasattr(optimizer, "state_dict")
                    else None,
                    "scheduler": scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else None,
                    "step": i,
                },
                f"ckpt-{str(i).zfill(6)}.pt",
            )

        if (
            i > 0
            and i % conf.evaluate.eval_freq == 0
            and not conf.evaluate.skip_evaluate
        ):
            report = evaluate(conf, mappers, valid_loader, model, device, logger)

            if dist.is_primary():
                checker.log(**report, step=i)


if __name__ == "__main__":
    conf = load_arg_config(E2EConfig, elastic=True)
    dist.run(conf, main, args=(conf,))
