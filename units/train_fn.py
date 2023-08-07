def wd_skip_fn(skip_type):
    def check_wd_skip_fn(name, param):
        if skip_type == "nfnet":
            return "bias" in name or "gain" in name

        elif skip_type == "resnet":
            return "bias" in name or "bn" in name or param.ndim == 1

        elif skip_type == "vit":
            return "bias" in name or "cls" in name or "norm" in name or param.ndim == 1

        elif skip_type == "dino":
            return "bias" in name or param.ndim == 1

    return check_wd_skip_fn


def add_weight_decay(named_parameters, weight_decay, check_skip_fn):
    decay = []
    decay_names = []
    no_decay = []
    no_decay_names = []
    lr_reduce = []
    lr_reduce_names = []

    for n, p in named_parameters:
        if not p.requires_grad:
            continue

        if check_skip_fn(n, p):
            no_decay.append(p)
            no_decay_names.append(n)

        else:
            if "sampling_offsets" in n:
                lr_reduce.append(p)
                lr_reduce_names.append(n)

            else:
                decay.append(p)
                decay_names.append(n)

    return (
        (
            {"params": no_decay, "weight_decay": 0.0, "no_decay": True},
            {"params": decay, "weight_decay": weight_decay},
            {
                "params": lr_reduce,
                "weight_decay": weight_decay,
                "lr": 3e-4 * 0.1,
                "weight_decay": weight_decay,
            },
        ),
        (no_decay_names, decay_names, lr_reduce_names),
    )


def sample_data(loader):
    while True:
        for data in loader:
            yield data
