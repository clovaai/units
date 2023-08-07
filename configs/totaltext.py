# sudo -H PYTHONPATH=$PWD python script/train.py --conf configs/finetune.py --ckpt weights/pretrain.pt
from tensorfn.config.builder import F, L, field
from tensorfn.nn import repeat
from torch import nn

from units.augment import (
    ColorJitter,
    ExpandPAD,
    MultiScaleCrop,
    RandomApply,
    RandomCrop,
    RandomGaussianBlur,
    RandomResizeScale,
    RandomRotate,
    RandomSelect,
    RandomTranspose,
    RandomUnsharpMask,
)
from units.datasource import LMDBSource
from units.models.backbones.swin_transformer import (
    swin_transformer_b,
    swin_transformer_s,
)
from units.models.data import UnitsMapper
from units.models.decoder import UnitsDecoder
from units.models.model import HybridTransformer, Units, VisionTransformer
from units.models.transformer import TransformerDecoder, TransformerDecoderLayer
from units.tokenizer import UnitsTokenizer
from units.transform import Compose, EdgeEnhance, Grayscale, Normalize, Resize, ToTensor

conf = field()
conf.finetune = True

# Common

exp_name = "totaltext"
data_path = "train_datasets"
prompt_type = "point"  # None, "roi", "order", "point"
tokenizer = UnitsTokenizer()
tokenizer.add_detection_vocab(bin_size=1000)
# tokenizer.add_order_vocab(max_order=150)
tokenizer.add_unify_annotation_vocab()
ignore_idx = -100
wo_source_pos_encod = True
stride16_feat = False
coord_order = "xy"


# Model

dim_enc = 512 if stride16_feat else 1024
dim_dec = 256
n_head_enc = 8
n_head_dec = 8
mlp_ratio_enc = 4
mlp_ratio_dec = 4
n_enc = 6
n_dec = 8
max_text_length = 25
n_object = 100
decoder_length = 1024

criterion = L[nn.CrossEntropyLoss](ignore_index=ignore_idx)

backbone = L[swin_transformer_b](pretrained=True)

encoder = L[VisionTransformer](
    backbone,
    (16,) if stride16_feat else (32,),
    wo_source_pos_encod,
)

trm_decoder = L[TransformerDecoder](
    repeat(L[TransformerDecoderLayer](dim_dec, n_head_dec, n_experts=2), n_dec),
    autoregressive=True,
)

decoder = L[UnitsDecoder](
    dim_dec,
    max_text_length,
    trm_decoder,
    criterion,
    n_object,
    decoder_length,
    tokenizer,
    prompt=prompt_type,
    detect_type="polygon",
    fixed_text_len=True,
    coord_order=coord_order,
    iterative_decoding=True,
)

conf.model = L[Units](
    dim_enc, dim_dec, encoder, decoder, wo_source_pos_encod=wo_source_pos_encod
)


# Training & Evaluate

mappers = [
    L[UnitsMapper](
        tokenizer,
        max_text_length,
        n_object=n_object,
        decoder_length=decoder_length,
        ignore_idx=ignore_idx,
        dcs_inputs=True,
        iou_filtering=False,
        all_unks_remove=False,
        coord_order=coord_order,
        fixed_text_len=True,
        prompt=prompt_type,
    )
]

# Training

angle = 45
train_size = 1920
valid_size = 1920
n_iter = 20_000

train_transform = [
    L[RandomSelect](
        [
            (L[RandomRotate](-angle, angle), 0.333),
        ],
    ),
    L[RandomResizeScale](1.0, 1.0, (train_size, train_size)),
    L[ExpandPAD]((train_size, train_size)),
    L[RandomApply](L[ColorJitter](0.4, 0.4, 0.2, 0.1), 0.8),
    L[RandomApply](L[Grayscale](), 0.3),
    L[RandomSelect](
        [
            (L[RandomGaussianBlur](), 0.3),
            # (L[RandomUnsharpMask](), 0.1),
            (L[EdgeEnhance](), 0.1),
            # (L[Compose]([L[RandomUnsharpMask](), L[EdgeEnhance]()]), 0.1),
        ],
    ),
    L[ToTensor](),
    L[Normalize](),
]

# mixed finetune
train_sets = [
    ("totaltext.poly_train.lmdb", 1.0),
]

datasources = field(datasource=F[LMDBSource](), path=data_path, sources=train_sets)

lr = 3e-5
weight_decay = 1e-4
wd_skip_fn = "vit"
batch_size = 16
num_workers = 4
val_batch_size = 16
val_num_workers = 4
clip_grad = 0.1

optimizer = field(type="adamw", lr=lr)
scheduler = field(
    type="constant",
)
loader = field(
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)
val_loader = field(
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=val_num_workers,
)
conf.training = field(
    n_iter=n_iter,
    datasources=datasources,
    transform=train_transform,
    mappers=mappers,
    optimizer=optimizer,
    scheduler=scheduler,
    loader=loader,
    val_loader=val_loader,
    weight_decay=weight_decay,
    wd_skip_fn=wd_skip_fn,
    clip_grad=clip_grad,
)

# Evaluate

valid_transform = [
    # L[RandomResizeScale](1.0, 1.0, (valid_size, valid_size)),
    L[Resize](valid_size),
    L[ExpandPAD]((valid_size, valid_size)),
    L[ToTensor](),
    L[Normalize](),
]

valid_sets = [
    "totaltext.poly_test.lmdb",
]

datasources = field(datasource=F[LMDBSource](), path=data_path, sources=valid_sets)

# skip evaluation
conf.evaluate = field(
    eval_metrics=[],
    eval_metrics_option=None,
    datasources=datasources,
    transform=valid_transform,
    eval_freq=5000,
    skip_evaluate=True,
)

conf.checker = field(
    storage=[field(type="local", path="checkpoints")],
    reporter=[
        field(type="logger"),
        field(
            type="wandb",
            project="units",
            name=exp_name,
        ),
    ],
)
