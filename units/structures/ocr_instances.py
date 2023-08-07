from typing import Set

from units.structures import Fields, InstanceType
from units.structures.polygons import Polygons


class OCRInstances(Fields):
    type: Set[InstanceType] = {InstanceType.COORDS}

    def __init__(
        self,
        polygons,
        texts,
        confidences=None,
    ):
        """
        !Important! text with length 0 ('') indicates don't care area!
        angles start from (1, 0) and rotates counter-clockwise
        """

        if len(polygons) != len(texts):
            raise ValueError(
                f"length of polygons and texts should be same; got {len(polygons)} vs {len(texts)}"
            )

        if not isinstance(polygons, Polygons):
            polygons = Polygons(polygons)

        self.polygons = polygons

        self.texts = self.get_fields(polygons, texts, "texts", "")
        self.confidences = self.get_fields(polygons, confidences, "confidences", 1)

    def get_fields(self, polygons, fields, field_names, default):
        if fields is not None:
            if len(polygons) != len(fields):
                raise ValueError(
                    (
                        f"length of polygons and {field_names} should be same;"
                        f" got {len(polygons)} vs {len(fields)}"
                    )
                )

            return fields

        else:
            return [default] * len(polygons)

    def __getitem__(self, key):
        return OCRInstances(
            self.polygons[key],
            self.texts[key],
            self.confidences[key],
        )

    @property
    def coords(self):
        return self.polygons.coords

    def resize(self, ratio_w, ratio_h):
        polygons = self.polygons.resize(ratio_w, ratio_h)

        return OCRInstances(
            polygons,
            self.texts,
            self.confidences,
        )

    def crop_and_resize(self, crop_x, crop_y, scale_w, scale_h, width, height):
        res_polygons, filtered = self.polygons.crop_and_resize(
            crop_x, crop_y, scale_w, scale_h, width, height
        )
        res_texts = []
        res_confidences = []

        for (filt, text, confidence,) in zip(
            filtered,
            self.texts,
            self.confidences,
        ):
            if filt:
                res_texts.append(text)
                res_confidences.append(confidence)

        return OCRInstances(
            res_polygons,
            res_texts,
            res_confidences,
        )

    def filter_dont_care(self, width, height):
        res_texts = []
        in_rects = self.polygons.check_in_rect(width, height)

        for in_rect, text in zip(in_rects, self.texts):
            if in_rect:
                res_texts.append(text)

            else:
                res_texts.append("")

        return OCRInstances(
            self.polygons,
            res_texts,
            self.confidences,
        )

    def rotate(self, angle, width, height):
        res_polygons, filtered = self.polygons.rotate(angle, width, height)
        res_texts = []
        res_confidences = []

        for filt, text, confidence in zip(
            filtered,
            self.texts,
            self.confidences,
        ):
            if filt:
                res_texts.append(text)
                res_confidences.append(confidence)

        return OCRInstances(
            res_polygons,
            res_texts,
            res_confidences,
        )

    def transpose(self, angle, width, height):
        return OCRInstances(
            self.polygons.transpose(angle, width, height),
            self.texts,
            self.confidences,
        )
