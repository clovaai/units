import torch

from units.transform_fn import affine_transform, get_affine_transform


class Polygons:
    def __init__(self, coords):
        self.coords = [torch.as_tensor(coord) for coord in coords]

    def __getitem__(self, key):
        if isinstance(key, int):
            return Polygons([self.coords[key]])

        return Polygons(self.coords[key])

    def __len__(self):
        return len(self.coords)

    def resize(self, ratio_w, ratio_h):
        ratio = torch.as_tensor((ratio_w, ratio_h)).unsqueeze(0)
        coords = [c * ratio for c in self.coords]

        return Polygons(coords)

    def _is_poly_in_rect(self, poly, x, y, w, h):
        if poly[:, 0].min().item() >= x and poly[:, 0].max().item() <= x + w:
            if poly[:, 1].min().item() >= y and poly[:, 1].max().item() <= y + h:
                return True

        return False

    def _is_poly_outside_rect(self, poly, x, y, w, h):
        if poly[:, 0].max().item() < x or poly[:, 0].min().item() > x + w:
            return True

        if poly[:, 1].max().item() < y or poly[:, 1].min().item() > y + h:
            return True

        return False

    def crop_and_resize(
        self, crop_x, crop_y, scale_w, scale_h, width=None, height=None
    ):
        offset = torch.tensor((crop_x, crop_y), dtype=torch.float32)
        scale = torch.tensor((scale_w, scale_h), dtype=torch.float32)
        coords = [(coord - offset) * scale for coord in self.coords]
        coords, filtered = self.filter_outside_rect(coords, width, height)

        return Polygons(coords), filtered

    def filter_outside_rect(self, coords, width, height):
        filtered_coords = []
        filtered = []

        for coord in coords:
            filt = not self._is_poly_outside_rect(coord, 0, 0, width, height)

            if filt:
                filtered_coords.append(coord)

            filtered.append(filt)

        return filtered_coords, filtered

    def check_in_rect(self, width, height):
        in_rects = []

        for coord in self.coords:
            in_rect = self._is_poly_in_rect(coord, 0, 0, width, height)
            in_rects.append(in_rect)

        return in_rects

    def rotate(self, angle, width, height):
        center = torch.tensor([width // 2, height // 2], dtype=torch.float32).numpy()
        size = torch.tensor((width, height), dtype=torch.float32).numpy()
        trans, _ = get_affine_transform(center, size, angle, size)
        trans = torch.as_tensor(trans, dtype=torch.float32)

        coords = []

        for coord in self.coords:
            coord = [affine_transform(p.tolist(), trans) for p in coord.unbind(0)]
            coord = torch.stack(coord, 0)
            coords.append(coord)

        coords, filtered = self.filter_outside_rect(coords, width, height)

        return Polygons(coords), filtered

    def transpose(self, angle, width, height):
        if angle == 0:
            return self

        w_half = width // 2
        h_half = height // 2

        if angle == 90:
            flip = True
            sign = torch.tensor((1, -1))
            shift = torch.tensor((-h_half + h_half, width))

        elif angle == -90:
            flip = True
            sign = torch.tensor((-1, 1))
            shift = torch.tensor((height, -w_half + w_half))

        elif abs(angle) == 180:
            flip = False
            sign = torch.tensor((-1, -1))
            shift = torch.tensor((width, height))

        else:
            raise ValueError("transpose only supports angle 90, -90, 180, -180")

        coord_transpose = []
        for coord in self.coords:
            if flip:
                x, y = coord.unbind(-1)
                coord = torch.stack((y, x), -1)

            coord = coord * sign + shift
            coord_transpose.append(coord)

        return Polygons(coord_transpose)
