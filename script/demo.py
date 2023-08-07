# PYTHONPATH=$PWD python script/demo.py --conf configs/finetune.py --ckpt weights/shared.pt

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tensorfn import load_arg_config
from tensorfn.config import instantiate

from units.config import E2EConfig
from units.dataset import MultitaskCollator
from units.structures import Sample
from units.transform import Compose

# for visualization
font = ImageFont.truetype("TC/Georgia.ttf", size=24)

DETECT_TYPE = "quad"  # 'single', 'box', 'quad', 'polygon'


def resize_instance(polygon, image_size, orig_size):
    """
    During training, the images were resized while maintaining the aspect ratio, and padding was added to fill the empty space.
    Thus, perform the inverse transformation.
    """
    ratio = max(orig_size) / max(image_size)

    return polygon * ratio


def draw_ocr(img, coords, texts, detect_type="quad", draw_width=5):
    ocr_img = img.copy()
    draw = ImageDraw.Draw(ocr_img)

    for coord in coords:
        if detect_type in ["quad", "polygon"]:
            coord = np.array(coord)
            draw.polygon(
                coord.reshape(-1).astype(np.int64).tolist(),
                outline="red",
                width=draw_width,
            )
        elif detect_type in ["single"]:
            x, y = coord[0]
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="red", width=draw_width)
        else:  # ['box']
            c1, c2 = coord
            x1, y1 = c1
            x2, y2 = c2
            draw.rectangle([x1, y1, x2, y2], outline="red", width=draw_width)

    for coord, text in zip(coords, texts):
        size = font.getsize(text)

        pos = coord[0].copy()
        pos[1] -= size[1]
        draw.rectangle(
            (pos[0] - 1, pos[1] - 1, pos[0] + size[0] + 1, pos[1] + size[1] + 1),
            fill=(0, 0, 0),
        )
        draw.text(pos, text, font=font, fill=(255, 255, 255, 255))

    return ocr_img


def run_ocr(img, model, detect_type):
    """
    Returns:
        coord_resize: [coord1, coord2, coord3, ...]
        texts: [text1, text2, text3, ...]
    """
    img = img.convert("RGB")
    o_w, o_h = img.size
    img, sample = transform(img, Sample(image_size=(o_h, o_w)))
    _, n_h, n_w = img.shape
    sample.image_size = (n_h, n_w)  # resized inputs

    batch = collator([(img, sample)])  # batch size 1

    with torch.inference_mode():
        batch = batch.to(device)
        out = model(batch, detect_type)

    # only the first batch is retrieved (since the batch size is forced to be 1).
    out = mapper.postprocess(batch, out)[0]

    coord_resize = torch.stack(out.coords, 0).numpy()
    coord_resize = resize_instance(coord_resize, (n_h, n_w), (o_h, o_w))
    coord_resize = coord_resize.round().astype(np.int64).tolist()

    return (
        coord_resize,
        out.texts,
    )


if __name__ == "__main__":
    # 1. model, mapper, transform, collator instantiate
    device = "cuda"
    conf = load_arg_config(E2EConfig, elastic=True, show=True)
    model = instantiate(conf.model).to(device)

    ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)
    ckpt_model = dict()
    for key, value in ckpt["model"].items():
        if conf.n_gpu > 1:
            ckpt_model[key] = value
        else:
            ckpt_model[key.replace("module.", "", 1)] = value

    model.load_state_dict(ckpt_model)
    model.eval()
    model = model.to(device)
    mapper = instantiate(conf.training.mappers)[0]
    transform = Compose(instantiate(conf.evaluate.transform))
    collator = MultitaskCollator([], evaluate=True)

    # 2. img load
    img_path = "TC/0000066.jpg"
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)

    # 3. run ocr
    coords, texts = run_ocr(img, model, DETECT_TYPE)

    # 4. visualization
    img_out = draw_ocr(img, coords, texts, DETECT_TYPE)
    img_out.save(f"result_{DETECT_TYPE}.jpg")
