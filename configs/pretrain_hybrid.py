# sudo -H PYTHONPATH=$PWD python script/train.py --conf configs/pretrain_hybrid.py
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
from units.models.backbones.resnet import resnet50
from units.models.data import UnitsMapper
from units.models.decoder import UnitsDecoder
from units.models.model import HybridTransformer, Units, VisionTransformer
from units.models.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from units.tokenizer import UnitsTokenizer
from units.transform import Compose, EdgeEnhance, Grayscale, Normalize, Resize, ToTensor

conf = field()

# Common

exp_name = "hybrid_pretrain"
data_path = "train_datasets"
prompt_type = "point"  # None, "roi", "order", "point"
tokenizer = UnitsTokenizer()
tokenizer.add_detection_vocab(bin_size=1000)
# tokenizer.add_order_vocab(max_order=150)
tokenizer.add_unify_annotation_vocab()
ignore_idx = -100
wo_source_pos_encod = False
coord_order = "xy"


# Model

dim_enc = 256
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

backbone = L[resnet50](pretrained=True, sync_bn=True, freeze_bn=False)

trm_encoder = L[TransformerEncoder](
    repeat(L[TransformerEncoderLayer](dim_enc, n_head_enc, mlp_ratio_enc), n_enc),
)
encoder = L[HybridTransformer](
    backbone,
    (16,),
    dim_enc,
    trm_encoder,
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
    detect_type="quad",
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
        iou_filtering=True,
        all_unks_remove=False,
        coord_order=coord_order,
        fixed_text_len=True,
        prompt=prompt_type,
        mixed_annot_change_prob=0.4,
        all_annot_change_prob=0.4,
    )
]

# Training

angle = 30
train_size_list = [640, 672, 704, 736, 768, 800, 832, 864, 896]
valid_size = 1280
n_iter = 300_000

train_transform = [
    L[RandomTranspose](0.03, 0.03, 0.03),
    L[RandomRotate](-angle, angle),
    L[MultiScaleCrop](0.3, 2.0, train_size_list),
    L[RandomApply](L[ColorJitter](0.4, 0.4, 0.2, 0.1), 0.8),
    L[RandomApply](L[Grayscale](), 0.3),
    L[RandomSelect](
        [
            (L[RandomGaussianBlur](), 0.3),
            (L[RandomUnsharpMask](), 0.1),
            (L[EdgeEnhance](), 0.1),
            (L[Compose]([L[RandomUnsharpMask](), L[EdgeEnhance]()]), 0.1),
        ],
    ),
    L[ToTensor](),
    L[Normalize](),
]

# large mixed pre-train
train_sets = [
    ("synthtext150k.poly.part1_train.lmdb", 30.0),
    ("synthtext150k.poly.part2_train.lmdb", 20.0),
    ("totaltext.poly_train.lmdb", 4.5),
    ("icdar13_train.lmdb", 1.0),
    ("icdar15_train.lmdb", 4.5),
    ("mlt19_train.lmdb", 10.0),
    ("hiertext_train.lmdb", 25.0),
    ("textocr_train.lmdb", 25.0),
]

datasources = field(datasource=F[LMDBSource](), path=data_path, sources=train_sets)

lr = 3e-4
weight_decay = 1e-4
wd_skip_fn = "vit"
batch_size = 64
num_workers = 8
val_batch_size = 64
val_num_workers = 8
clip_grad = 0.1

optimizer = field(type="adamw", lr=lr)
scheduler = field(
    type="cycle",
    lr=lr,
    n_iter=n_iter + 1,
    initial_multiplier=1e-5,
    warmup=10000,
    decay=("linear", "cos"),
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
    L[Resize](valid_size),
    L[ToTensor](),
    L[Normalize](),
]

valid_sets = [
    "totaltext.poly_test.lmdb",
    "icdar15_test.lmdb",
    # "textocr_val.lmdb",
    # "ctw1500.poly_test.lmdb",
]

datasources = field(datasource=F[LMDBSource](), path=data_path, sources=valid_sets)

# skip evaluation
conf.evaluate = field(
    eval_metrics=[],
    eval_metrics_option=None,
    datasources=datasources,
    transform=valid_transform,
    eval_freq=25000,
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
