import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as tfn

from units.structures import InstanceType


class Transform:
    def get_params(self, img, sample):
        return {}

    def apply_img(self, img, params):
        return img

    def apply_coords(self, coords, params):
        return coords

    def __call__(self, img, sample):
        params = self.get_params(img, sample)

        img = self.apply_img(img, params)

        for k, v in sample.fields().items():
            if hasattr(v, "type") and InstanceType.COORDS in v.type:
                v = self.apply_coords(v, params)
                sample.set(k, v)

        return img, sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, sample):
        for transform in self.transforms:
            img, sample = transform(img, sample)

        return img, sample


class Normalize(Transform):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def apply_img(self, img, params):
        return tfn.normalize(img, self.mean, self.std)

    def denormalize(self, img):
        mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device)
        std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device)

        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)

        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return img * std + mean


class ToTensor(Transform):
    def apply_img(self, img, params):
        return tfn.to_tensor(img)


class Resize(Transform):
    def __init__(self, max_size, interpolation=Image.BILINEAR):
        self.max_size = max_size
        self.interp = interpolation

    def get_size(self, img, max_size):
        w, h = img.size

        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = int(max_size * short / long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)

        return new_w, new_h

    def get_params(self, img, sample):
        ow, oh = img.size
        nw, nh = self.get_size(img, self.max_size)
        rw, rh = nw / ow, nh / oh

        return {"nw": nw, "nh": nh, "rw": rw, "rh": rh}

    def apply_img(self, img, params):
        return img.resize((params["nw"], params["nh"]), self.interp)

    def apply_coords(self, coords, params):
        return coords.resize(params["rw"], params["rh"])


class Grayscale(Transform):
    def apply_img(self, img, params):
        return img.convert("L").convert("RGB")


class EdgeEnhance(Transform):
    def apply_img(self, img, params):
        return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
