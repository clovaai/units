# PYTHONPATH=$PWD python script/test.py --conf configs/finetune.py --ckpt weights/shared.pt

import warnings

warnings.filterwarnings("ignore")
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from tensorfn import distributed as dist
from tensorfn import get_logger, load_arg_config
from tensorfn.config import instantiate
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from units.config import E2EConfig
from units.dataset import MultitaskCollator, MultitaskDataset, WeightedDataset
from units.transform import Compose

# config
PRED_PATH = "res_output"
# DETECT_TYPE = "quad"  # 'single', 'box', 'quad', 'polygon'
font = ImageFont.truetype("TC/Georgia.ttf", size=24)


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
def predict(conf, mappers, loader, model, device, logger):
    is_train = model.training
    model.eval()

    # metrics = Metrics(
    #     conf.evaluate.eval_metrics, mappers, device, conf.evaluate.eval_metrics_option
    # )
    total = len(loader)

    if os.path.isdir(PRED_PATH):
        os.system("rm -r {}".format(PRED_PATH))
    os.system("mkdir {}".format(PRED_PATH))

    for i, batch in enumerate(loader):
        # out = model(batch.to(device), DETECT_TYPE)
        # The DETECT_TYPE can also be adjusted in the configuration file.
        out = model(batch.to(device))
        # metrics(batch, out)
        pred_instances = mappers[0].postprocess(batch, out)

        for batch_i in range(len(batch.samples)):
            filename = batch.samples[batch_i].img_path.split("/")[-1].split(".")[0]

            pred = pred_instances[batch_i]
            coords = [coord.cpu().numpy() for coord in pred.coords]
            texts = [text for text in pred.texts]
            scores = pred.confidences

            # normalization
            ratio = max(batch.samples[batch_i].orig_size) / max(
                batch.samples[batch_i].image_size
            )
            coords = [coord * ratio for coord in coords]
            coords = [
                list(map(str, map(int, coord.reshape(-1).tolist()))) for coord in coords
            ]

            with open(os.path.join(PRED_PATH, "res_" + filename + ".txt"), "w") as file:
                for coord, text, score in zip(coords, texts, scores):
                    # coord = ['%.2f' % elem for elem in coord]
                    message = ",".join(coord) + f",{score},{text}"
                    print(filename)
                    print(message)
                    file.write(message + "\n")

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

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
            find_unused_parameters=True,
        )

    mappers = instantiate(conf.training.mappers)

    # when evaluate, we don't want to skip invalid sample.
    for mapper in mappers:
        mapper.skip_invalid_sample = False

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

    checker = instantiate(conf.checker)
    checker.catalog(conf)

    ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)
    ckpt_model = dict()
    for key, value in ckpt["model"].items():
        if conf.n_gpu > 1:
            ckpt_model[key] = value
        else:
            ckpt_model[key.replace("module.", "", 1)] = value

    model.load_state_dict(ckpt_model)
    predict(conf, mappers, valid_loader, model, device, logger)


if __name__ == "__main__":
    conf = load_arg_config(E2EConfig, elastic=True)
    dist.run(conf, main, args=(conf,))
