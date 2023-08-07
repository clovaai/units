from typing import Dict, List, Optional, Tuple, Union

from pydantic import StrictBool, StrictInt, StrictStr
from tensorfn.config import (
    Checker,
    Config,
    DataLoader,
    Instance,
    MainConfig,
    Optimizer,
    Scheduler,
)


class DataSources(Config):
    datasource: Instance
    path: StrictStr
    sources: List[Union[StrictStr, Tuple[StrictStr, float]]]


class Training(Config):
    n_iter: int

    datasources: Union[DataSources, List[DataSources]]
    mappers: List[Instance]
    transform: List[Instance]
    img_multiple: int = 32

    optimizer: Optimizer
    weight_decay: float = 0
    wd_skip_fn: StrictStr = "vit"
    clip_grad: float = 0
    scheduler: Scheduler
    loader: DataLoader
    val_loader: DataLoader

    resume_ckpt_freq: StrictInt = 100
    mixed_precision: bool = False


class Evaluate(Config):
    eval_freq: StrictInt = 10000
    eval_metrics: List[StrictStr]
    eval_metrics_option: Optional[Dict] = None

    datasources: DataSources
    transform: List[Instance]

    skip_evaluate: StrictBool = False


class E2EConfig(MainConfig):
    model: Instance
    training: Training
    evaluate: Evaluate
    finetune: StrictBool = False

    log_freq: StrictInt = 10
    checker: Checker = Checker()
    logger: StrictStr = "rich"
