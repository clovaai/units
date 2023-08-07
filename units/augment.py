import random

from PIL import Image, ImageFilter
from torchvision import transforms as tfms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_fn
from torchvision.transforms.functional import pil_modes_mapping

from units.transform import Transform


class RandomApply(Transform):
    def __init__(self, transform, p):
        self.transform = transform
        self.p = p

    def get_params(self, img, sample):
        apply = random.random() < self.p
        params = self.transform.get_params(img, sample)

        return {"apply": apply, "params": params}

    def apply_img(self, img, params):
        if not params["apply"]:
            return img

        return self.transform.apply_img(img, params["params"])

    def apply_coords(self, coords, params):
        if not params["apply"]:
            return coords

        return self.transform.apply_coords(coords, params["params"])


class RandomSelect:
    def __init__(self, transforms):
        probs = [tfms[1] for tfms in transforms]
        none_prob = 1 - sum(probs)
        self.probs = [none_prob] + probs
        self.transforms = [None] + [tfms[0] for tfms in transforms]

    def __call__(self, img, sample):
        selected = random.choices(self.transforms, self.probs, k=1)[0]

        if selected is None:
            return img, sample

        return selected(img, sample)


class RandomGaussianBlur(Transform):
    def __init__(self, radius_min=0.1, radius_max=2):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def get_params(self, img, sample):
        return {"radius": random.uniform(self.radius_min, self.radius_max)}

    def apply_img(self, img, params):
        return img.filter(ImageFilter.GaussianBlur(radius=params["radius"]))


class RandomUnsharpMask(Transform):
    def __init__(self, radius_min=1, radius_max=5, percent_min=150, percent_max=500):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.percent_min = percent_min
        self.percent_max = percent_max

    def get_params(self, img, sample):
        return {
            "radius": random.uniform(self.radius_min, self.radius_max),
            "percent": random.randint(self.percent_min, self.percent_max),
        }

    def apply_img(self, img, params):
        return img.filter(ImageFilter.UnsharpMask(params["radius"], params["percent"]))


class ColorJitter(Transform):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.transform = tfms.ColorJitter(brightness, contrast, saturation, hue)

    def apply_img(self, img, params):
        return self.transform(img)


class RandomRotate(Transform):
    def __init__(self, low, high, interpolation=InterpolationMode.BILINEAR):
        self.low = low
        self.high = high
        self.resample = pil_modes_mapping[interpolation]

    def get_params(self, img, sample):
        angle = random.uniform(self.low, self.high)
        width, height = img.size

        return {"angle": angle, "width": width, "height": height}

    def apply_img(self, img, params):
        return img.rotate(params["angle"], resample=self.resample)

    def apply_coords(self, coords, params):
        return coords.rotate(params["angle"], params["width"], params["height"])


class RandomTranspose(Transform):
    def __init__(self, prob_90, prob_180, prob_270):
        if not ((0 <= prob_90 <= 1) and (0 <= prob_180 <= 1) and (0 <= prob_270 <= 1)):
            raise ValueError("Probability of transpose angles should be in [0, 1]")

        if prob_90 + prob_180 + prob_270 > 1:
            raise ValueError("Sum of proabiblities of each angels should be <= 1")

        self.choices = (
            (None, None),
            (90, Image.ROTATE_90),
            (180, Image.ROTATE_180),
            (-90, Image.ROTATE_270),
        )
        self.probs = (1 - prob_90 - prob_180 - prob_270, prob_90, prob_180, prob_270)

    def get_params(self, img, sample):
        angle, transpose = random.choices(self.choices, weights=self.probs, k=1)[0]
        width, height = img.size

        return {
            "angle": angle,
            "transpose": transpose,
            "width": width,
            "height": height,
        }

    def apply_img(self, img, params):
        if params["transpose"] is None:
            return img

        return img.transpose(params["transpose"])

    def apply_coords(self, coords, params):
        if params["angle"] is None:
            return coords

        return coords.transpose(params["angle"], params["width"], params["height"])


class RandomResizeScale(Transform):
    def __init__(
        self,
        min_ratio,
        max_ratio,
        target_size,
        interpolation=InterpolationMode.BILINEAR,
    ):
        self.min_size = min_ratio
        self.max_size = max_ratio
        self.target_size = target_size
        self.interpolation = interpolation

    def get_size(self, img_size):
        w, h = img_size
        size = random.uniform(self.min_size, self.max_size)
        ow, oh = self.target_size[0] * size, self.target_size[1] * size
        scale = min(ow / w, oh / h)
        ow = int(w * scale)
        oh = int(h * scale)

        return ow, oh

    def get_params(self, img, sample):
        size_orig = img.size
        size = self.get_size(img.size)
        rw, rh = size[0] / size_orig[0], size[1] / size_orig[1]

        return {"nw": size[0], "nh": size[1], "rw": rw, "rh": rh}

    def apply_img(self, img, params):
        return transform_fn.resize(
            img, (params["nh"], params["nw"]), interpolation=self.interpolation
        )

    def apply_coords(self, coords, params):
        return coords.resize(params["rw"], params["rh"])


class MultiScaleCrop(Transform):
    def __init__(
        self,
        min_ratio,
        max_ratio,
        target_size_list,
        interpolation=InterpolationMode.BILINEAR,
    ):
        self.min_size = min_ratio
        self.max_size = max_ratio
        self.target_size_list = target_size_list
        self.interpolation = interpolation

    def get_size(self, img_size, target_w, target_h):
        w, h = img_size
        size = random.uniform(self.min_size, self.max_size)
        ow, oh = target_w * size, target_h * size
        scale = min(ow / w, oh / h)
        ow = int(w * scale)
        oh = int(h * scale)

        return ow, oh

    def get_params(self, img, sample):
        # multi-scale resize
        size_orig = img.size
        target_size = random.choice(
            self.target_size_list
        )  # TODO: square -> extend non-square
        size = self.get_size(img.size, target_size, target_size)
        rw, rh = size[0] / size_orig[0], size[1] / size_orig[1]

        # crop
        in_w, in_h = size
        out_w, out_h = target_size, target_size

        crop_x = random.randint(0, max(0, in_w - out_w))
        crop_y = random.randint(0, max(0, in_h - out_h))
        crop_w = min(crop_x + out_w, in_w) - crop_x
        crop_h = min(crop_y + out_h, in_h) - crop_y

        scale = 1
        h = int(crop_h * scale)
        w = int(crop_w * scale)

        return {
            "nw": size[0],
            "nh": size[1],
            "rw": rw,
            "rh": rh,
            "scale": 1,
            "crops": (crop_x, crop_y, crop_w, crop_h),
            "h": h,
            "w": w,
        }

    def apply_img(self, img, params):
        # multi-scale resize
        resized_img = transform_fn.resize(
            img, (params["nh"], params["nw"]), interpolation=self.interpolation
        )

        # crop
        c_x, c_y, c_w, c_h = params["crops"]
        return resized_img.crop((c_x, c_y, c_x + c_w, c_y + c_h))

    def apply_coords(self, coords, params):
        resized_coords = coords.resize(params["rw"], params["rh"])

        c_x, c_y, _, _ = params["crops"]
        return resized_coords.crop_and_resize(
            c_x, c_y, params["scale"], params["scale"], params["w"], params["h"]
        )


class RandomCrop(Transform):
    def __init__(
        self,
        size=(640, 640),
    ):
        self.size = size

    def get_params(self, img, sample):
        in_w, in_h = img.size
        out_w, out_h = self.size

        crop_x = random.randint(0, max(0, in_w - out_w))
        crop_y = random.randint(0, max(0, in_h - out_h))
        crop_w = min(crop_x + out_w, in_w) - crop_x
        crop_h = min(crop_y + out_h, in_h) - crop_y

        scale = 1
        h = int(crop_h * scale)
        w = int(crop_w * scale)

        return {
            "scale": 1,
            "crops": (crop_x, crop_y, crop_w, crop_h),
            "h": h,
            "w": w,
        }

    def apply_img(self, img, params):
        c_x, c_y, c_w, c_h = params["crops"]

        img = img.crop((c_x, c_y, c_x + c_w, c_y + c_h))

        return img

    def apply_coords(self, coords, params):
        c_x, c_y, _, _ = params["crops"]
        return coords.crop_and_resize(
            c_x, c_y, params["scale"], params["scale"], params["w"], params["h"]
        )


class ExpandPAD(Transform):
    def __init__(
        self,
        size=(640, 640),
        pad_color=(128, 128, 128),
    ):
        self.size = size
        self.pad_color = pad_color

    def apply_img(self, img, params):
        in_w, in_h = img.size
        out_w, out_h = self.size

        assert out_w >= in_w and out_h >= in_h

        result = Image.new(img.mode, (out_w, out_h), self.pad_color)
        result.paste(img, (0, 0))

        return result

    def apply_coords(self, coords, params):
        return coords
