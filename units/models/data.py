"""
UNITS
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import math
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from units.dataset import SampleError
from units.models.utils import (
    make_duplicate_coord,
    make_random_coord,
    make_random_tokens,
    make_shift_coord,
    poly_center,
)
from units.structures import FieldsMixin, OCRInstances


class DetectFormat:
    def __init__(self, num_pts, detect_type, token):
        self.num_pts = num_pts
        self.detect_type = detect_type
        self.token = token


@dataclass
class UnitsSample:
    class_ids: torch.Tensor  # 0: noise, 1: text, -1: dcs
    texts: torch.Tensor
    coords: List[torch.Tensor]
    detect_types: List[int]  # 0: single, 1: box, 2: quad, 3: polygon
    # recogn_types: torch.Tensor # 0: case-sensitive, 1: case-insensitive
    width: int
    height: int
    prompt_input: List[int]
    """
    roi: [tlx, tly, brx, bry]
    order: [start_idx, end_idx]
    point: [x, y]
    """


@dataclass
class UnitsBatch(FieldsMixin):
    units_inputs: torch.Tensor
    units_targets: torch.Tensor
    tasks: torch.Tensor  # detect -> 0, recognition -> 1
    w: torch.Tensor
    h: torch.Tensor


class UnitsMapper:
    name: str = "units"
    task: str = "ocr"

    def __init__(
        self,
        tokenizer,
        max_text_length,
        n_object=100,
        decoder_length=1024,
        img_multiple=32,
        ignore_idx=-100,
        drop_token_prob=0.0,
        eos_only_end=False,
        dcs_inputs=True,
        iou_filtering=True,
        boundary_clipping=True,
        all_unks_remove=True,
        w_noise_augmentation=False,
        coord_order="xy",
        fixed_text_len=True,
        text_order_type="serialization",
        permutation=False,
        input_perturbation=False,
        prompt=None,
        mixed_annot_change_prob=0.0,
        all_annot_change_prob=0.0,
        skip_invalid_sample=True,
    ):
        """
        Args:
            tokenizer: tokenizer for units
            max_text_length (int): maximum length of text (word transcription)
            n_object (int): the maximum number of objects (including augmented noise)
            decoder_length (int): decoder max length
            ignore_idx (float): ignore index of targets
            drop_token_prob (float): drop token ratio for input (text transcription masking)
            eos_only_end (bool): eos appears only at the end or several times on the augmentation noise
                                 if True => noise target is [na, na, na, na, noise] (pix2seq official code version)
                                 False => noise target is [eos, na, na, na, noise] (pix2seq official paper version)
            dcs_inputs (bool): whether input sequence includes dcs or not
            iou_filtering (bool): cropped instance filtering by box iou threshold
            boundary_clipping: boundary region clipping
            all_unks_remove: remove all unks samples
            w_noise_augmentation (bool): whether augment noise in object sequence or not
            coord_order (string): 'xy' or 'yx'
            fixed_text_len (bool): fixed transcription length (including [pad]) or not
            text_order_type: text ordering method
                            None => using raw data
                            serialization (string) => sorting by topleft serialization method
                            random (string) => random shuffling
            permutation (bool): whether apply permutation into the order of objects or not
            input_perturbation (bool): whether insert input coordinate noise
            prompt: decoder input prompt type (roi or order)
                    None => without prompt
                    roi (string) => using roi prompt
                    order (string) => using order span prompt
            skip_invalid_sample: whether skip invalid sample or not
        """
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.n_object = n_object
        self.decoder_length = decoder_length
        self.img_multiple = img_multiple
        self.ignore_idx = ignore_idx

        self.drop_token_prob = drop_token_prob

        self.eos_only_end = eos_only_end
        self.dcs_inputs = dcs_inputs
        self.iou_filtering = iou_filtering
        self.boundary_clipping = boundary_clipping
        self.all_unks_remove = all_unks_remove
        self.w_noise_augmentation = w_noise_augmentation
        self.fixed_text_len = fixed_text_len

        self.coord_order = coord_order
        assert coord_order in ["xy", "yx"]

        if text_order_type is not None:
            assert text_order_type in ["serialization", "random"]
        self.text_order_type = text_order_type
        self.permutation = permutation
        self.input_perturbation = input_perturbation

        # special tokens index (input mask, noise, special prompts, coord outside image)
        self.mask_token = self.tokenizer.vocab["[mask]"]
        self.noise_token = self.tokenizer.vocab["[noise]"]
        # self.text_token = self.tokenizer.vocab["[text]"] # for detect only
        self.roi_token = self.tokenizer.vocab["[roi]"]
        self.order_token = self.tokenizer.vocab["[order]"]
        self.point_token = self.tokenizer.vocab["[point]"]
        self.coord_out_token = self.tokenizer.vocab["[coord-out]"]

        self.prompt = prompt
        if prompt in ["order", "point"]:
            assert self.w_noise_augmentation == False

        self.mixed_annot_change_prob = mixed_annot_change_prob
        self.all_annot_change_prob = all_annot_change_prob

        self.single_format = DetectFormat(num_pts=1, detect_type=0, token="[single]")
        self.box_format = DetectFormat(num_pts=2, detect_type=1, token="[box]")
        self.quad_format = DetectFormat(num_pts=4, detect_type=2, token="[quad]")
        self.polygon_format = DetectFormat(num_pts=16, detect_type=3, token="[polygon]")

        self.detect_formats = [
            self.single_format,
            self.box_format,
            self.quad_format,
            self.polygon_format,
        ]

        self.detect_task_id = 0
        self.recog_task_id = 1

        self.skip_invalid_sample = skip_invalid_sample

    def __call__(self, sample):
        img_h, img_w = sample.image_size

        ocr = sample.ocr.filter_dont_care(img_w, img_h)

        coords = [coord.numpy() for coord in ocr.coords]
        texts = [text for text in ocr.texts]

        # Filtering
        if self.iou_filtering:
            coords, texts = self._iou_filtering(coords, texts, img_w, img_h)

        if self.boundary_clipping:
            coords = self._boundary_clipping(coords, img_w, img_h)

        coords = self._refine_coords(coords)

        # Serialization & permutation
        instance_ids = self._serialize_text(coords, self.text_order_type)

        if self.permutation:
            instance_ids = self._apply_permutation(instance_ids)

        texts = [texts[i] for i in instance_ids]
        coords = [coords[i] for i in instance_ids]

        detect_types = self._determine_detect_types(coords)

        # Change detection formats
        if np.random.random() <= self.mixed_annot_change_prob:
            # Change each instances to randomly selected detection format
            coords, detect_types = self._convert_random_detect_types(
                coords, detect_types, annot_change_prob=0.5
            )
        else:
            # Change all instances to same detection format
            if np.random.random() <= self.all_annot_change_prob:
                if np.random.random() <= 0.5:
                    coords, detect_types = self._convert_only_single_type(
                        coords, detect_types
                    )
                else:
                    coords, detect_types = self._convert_only_box_type(
                        coords, detect_types
                    )

        if self.all_unks_remove:
            coords, texts, detect_types = self._remove_all_unks(
                coords, texts, detect_types
            )

        # Select prompt input & extract instances corresponding to prompt input
        if self.prompt == "roi":
            (
                prompt_input,
                coords,
                texts,
                detect_types,
                is_valid_sample,
            ) = self._sampling_roi(
                coords,
                texts,
                detect_types,
                img_w,
                img_h,
            )
        elif self.prompt == "order":
            prompt_input, coords, texts, detect_types = self._sampling_orders(
                coords,
                texts,
                detect_types,
                self.tokenizer.max_order,
                self.dcs_inputs,
                self.n_object,
            )
        elif self.prompt == "point":
            (
                prompt_input,
                coords,
                texts,
                detect_types,
                is_valid_sample,
            ) = self._sampling_starting_point(
                coords,
                texts,
                detect_types,
                img_w,
                img_h,
            )
        else:
            prompt_input = None

        ignores = []
        for i in range(len(coords)):
            string = texts[i]

            if string == "":
                # Ignore empty text (Don't care, ###, invisible text)
                ignores.append(True)
            else:
                ignores.append(False)

        candid_texts = []
        candid_coords = []
        candid_detect_types = []
        candid_dcs = []

        for coord, text, detect_type, ignore in zip(
            coords, texts, detect_types, ignores
        ):
            if not self.dcs_inputs and ignore:
                continue

            tokens = self.tokenizer(text)

            candid_texts.append(tokens)
            candid_coords.append(coord.copy())
            candid_detect_types.append(detect_type)
            candid_dcs.append(ignore)

        if self.skip_invalid_sample:
            if self.prompt == None and len(candid_texts) < 1:
                raise SampleError("No valid text instance found")
            elif self.prompt in ["roi", "point"] and not is_valid_sample:
                raise SampleError(f"No valid {self.prompt}")

        (
            selected_texts,
            selected_coords,
            selected_categories,
            selected_detect_types,
        ) = self._make_sequence_for_ocr(
            candid_texts,
            candid_coords,
            candid_detect_types,
            candid_dcs,
            img_h,
            img_w,
            prompt_input,
        )

        selected_categories = torch.tensor(selected_categories)
        selected_texts = [torch.tensor(text) for text in selected_texts]
        selected_coords = [torch.tensor(coords) for coords in selected_coords]

        return UnitsSample(
            selected_categories,
            selected_texts,
            selected_coords,
            selected_detect_types,
            img_w,
            img_h,
            prompt_input,
        )

    def postprocess(self, batch_samples, outputs):
        batched_vertices = outputs["vertices"]
        batched_texts = outputs["texts"]
        batched_scores = outputs["scores"]
        instances = []

        for batch_vertice, batch_text, batch_score in zip(
            batched_vertices, batched_texts, batched_scores
        ):
            if batch_vertice is None:
                instances.append(OCRInstances([], []))
                continue

            texts = [self.tokenizer.decode(text) for text in batch_text]

            output_vertices, output_texts, output_scores = [], [], []
            for i, text in enumerate(texts):
                if "[mask]" not in text:
                    output_vertices.append(batch_vertice[i].tolist())
                    output_texts.append(text)
                    output_scores.append(batch_score[i])
            ocr_instances = OCRInstances(output_vertices, output_texts, output_scores)
            instances.append(ocr_instances)

        return instances

    def _get_batch_shape(self, batch):
        max_height = 0
        max_width = 0
        max_text_len = 0

        for b in batch:
            max_height = max(max_height, b.units.height)
            max_width = max(max_width, b.units.width)

            for s in b.units.texts:
                max_text_len = max(max_text_len, len(s))

        return max_height, max_width, max_text_len

    def collate_fn(self, batch):
        batch_size = len(batch)
        height, width, _ = self._get_batch_shape(batch)

        height = math.ceil(height / self.img_multiple) * self.img_multiple
        width = math.ceil(width / self.img_multiple) * self.img_multiple

        h = torch.zeros(batch_size, dtype=torch.float32)
        w = torch.zeros(batch_size, dtype=torch.float32)

        sequence_length = self.decoder_length
        batched_sequence_inputs = torch.zeros(
            batch_size, sequence_length, dtype=torch.int64
        )
        batched_sequence_targets = torch.zeros(
            batch_size, sequence_length, dtype=torch.int64
        )
        batched_sequence_tasks = torch.zeros(
            batch_size, sequence_length, dtype=torch.int64
        )

        for i, sample_i in enumerate(batch):
            sample_i = sample_i.units
            n_object = sample_i.class_ids.shape[0]
            class_ids = sample_i.class_ids.tolist()
            h[i], w[i] = sample_i.height, sample_i.width

            sample_i_sequence_inputs = []
            sample_i_seqeunce_targets = []
            sample_i_sequence_tasks = []

            # Convert coordinate to bin (quantization)
            for j in range(n_object):
                object_category = class_ids[j]
                object_coord = sample_i.coords[j].numpy()
                object_text = sample_i.texts[j].tolist()

                (
                    object_j_inputs,
                    object_j_targets,
                    object_j_tasks,
                ) = self._make_object_sequence(
                    object_category,
                    object_coord,
                    sample_i.width,
                    sample_i.height,
                    object_text,
                    sample_i.detect_types[j],
                )
                sample_i_sequence_inputs.extend(object_j_inputs)
                sample_i_seqeunce_targets.extend(object_j_targets)
                sample_i_sequence_tasks.extend(object_j_tasks)

            last_detect_type = self._determine_last_detect_type(
                sample_i_sequence_inputs
            )
            last_detect_token = self._determine_detect_token(last_detect_type)

            sample_i_sequence_inputs.append(last_detect_token)
            sample_i_seqeunce_targets.append(self.ignore_idx)
            sample_i_sequence_tasks.append(self.detect_task_id)

            sample_i_sequence_inputs.insert(0, self.tokenizer.go)
            sample_i_seqeunce_targets.append(self.tokenizer.eos)
            sample_i_sequence_tasks.append(self.detect_task_id)

            if self.prompt == "roi":
                (
                    sample_i_sequence_inputs,
                    sample_i_seqeunce_targets,
                    sample_i_sequence_tasks,
                ) = self._add_roi_point_prompt(
                    sample_i_sequence_inputs,
                    sample_i_seqeunce_targets,
                    sample_i_sequence_tasks,
                    sample_i.prompt_input,
                )
            elif self.prompt == "order":
                (
                    sample_i_sequence_inputs,
                    sample_i_seqeunce_targets,
                    sample_i_sequence_tasks,
                ) = self._add_order_prompt(
                    sample_i_sequence_inputs,
                    sample_i_seqeunce_targets,
                    sample_i_sequence_tasks,
                    sample_i.prompt_input,
                )
            elif self.prompt == "point":
                (
                    sample_i_sequence_inputs,
                    sample_i_seqeunce_targets,
                    sample_i_sequence_tasks,
                ) = self._add_point_prompt(
                    sample_i_sequence_inputs,
                    sample_i_seqeunce_targets,
                    sample_i_sequence_tasks,
                    sample_i.prompt_input,
                )

            if len(sample_i_sequence_inputs) < sequence_length:
                sample_i_sequence_inputs += (
                    sequence_length - len(sample_i_sequence_inputs)
                ) * [self.tokenizer.eos]

            if len(sample_i_seqeunce_targets) < sequence_length:
                sample_i_seqeunce_targets += (
                    sequence_length - len(sample_i_seqeunce_targets)
                ) * [
                    self.tokenizer.eos
                ]  # [self.ignore_idx]

            sample_i_sequence_inputs = sample_i_sequence_inputs[:sequence_length]
            sample_i_seqeunce_targets = sample_i_seqeunce_targets[:sequence_length]
            sample_i_sequence_tasks = sample_i_sequence_tasks[:sequence_length]

            sample_i_sequence_inputs = torch.Tensor(sample_i_sequence_inputs)
            sample_i_seqeunce_targets = torch.Tensor(sample_i_seqeunce_targets)
            sample_i_sequence_tasks = torch.Tensor(sample_i_sequence_tasks)

            batched_sequence_inputs[
                i, : sample_i_sequence_inputs.shape[0]
            ] = sample_i_sequence_inputs
            batched_sequence_targets[
                i, : sample_i_seqeunce_targets.shape[0]
            ] = sample_i_seqeunce_targets
            batched_sequence_tasks[
                i, : sample_i_sequence_tasks.shape[0]
            ] = sample_i_sequence_tasks

        return UnitsBatch(
            batched_sequence_inputs,
            batched_sequence_targets,
            batched_sequence_tasks,
            w,
            h,
        )

    def _iou_filtering(self, coords, texts, img_w, img_h, threshold=0.6):
        """
        Remove invalid instances whose overlap region is small.
        """
        refined_coords, refined_texts = [], []
        img_box = (0, 0, img_w, img_h)

        for coord, text in zip(coords, texts):
            text_box = np.min(coord, axis=0).tolist() + np.max(coord, axis=0).tolist()
            if (
                text_box[0] < 0
                or text_box[1] < 0
                or text_box[2] > img_w
                or text_box[3] > img_h
            ):
                text_box_area = (text_box[2] - text_box[0]) * (
                    text_box[3] - text_box[1]
                )

                x1 = max(text_box[0], img_box[0])
                y1 = max(text_box[1], img_box[1])
                x2 = min(text_box[2], img_box[2])
                y2 = min(text_box[3], img_box[3])

                intersect_w = max(0, x2 - x1)
                intersect_h = max(0, y2 - y1)

                inter = intersect_w * intersect_h
                iou = inter / text_box_area

                if iou < threshold:
                    continue

            refined_coords.append(coord)
            refined_texts.append(text)

        return refined_coords, refined_texts

    def _boundary_clipping(self, coords, img_w, img_h):
        """
        Clip text region cropped from image boundary, and convert to bounding box.
        """
        for i, coord in enumerate(coords):
            tlx, tly = np.min(coord, axis=0)
            brx, bry = np.max(coord, axis=0)
            if tlx < 0 or tly < 0 or brx >= img_w or bry >= img_h:
                tlx = np.clip(tlx, 0, img_w - 1)
                tly = np.clip(tly, 0, img_h - 1)
                brx = np.clip(brx, 0, img_w - 1)
                bry = np.clip(bry, 0, img_h - 1)
                coords[i] = np.array([[tlx, tly], [brx, bry]], dtype=np.float32)
        return coords

    def _refine_coords(self, coords, prob_box=0.5):
        """
        Convert to bounding box or single point for invalid annotation.
        """
        for i, coord in enumerate(coords):
            if coord.shape[0] not in [
                detection_format.num_pts for detection_format in self.detect_formats
            ]:
                if np.random.random() <= prob_box:
                    tlx, tly = np.min(coord, axis=0)
                    brx, bry = np.max(coord, axis=0)
                    coords[i] = np.array([[tlx, tly], [brx, bry]], dtype=np.float32)
                else:
                    coords[i] = poly_center(coord)
        return coords

    def _serialize_text(self, coords, text_order_type):
        """
        Serialize text instances.
        """
        instance_ids = list(range(len(coords)))
        if text_order_type == "serialization":
            instance_ids.sort(key=lambda x: poly_center(coords[x])[0].tolist()[::-1])
        elif self.text_order_type == "random":
            random.shuffle(instance_ids)
        return instance_ids

    def _remove_all_unks(self, coords, texts, detect_types):
        """
        Remove text with all unks.
        """
        refined_coords, refined_texts, refined_detect_types = [], [], []
        for coord, text, detect_type in zip(coords, texts, detect_types):
            tokens = self.tokenizer(text)
            if set(tokens) == {self.tokenizer.unk}:
                continue

            refined_coords.append(coord)
            refined_texts.append(text)
            refined_detect_types.append(detect_type)
        return refined_coords, refined_texts, refined_detect_types

    def _apply_permutation(self, inputs, shuffle_ratio=0.1, neighbor_span=1):
        """
        Permute sequence order.
        """
        for idx in range(len(inputs) - neighbor_span):
            if np.random.random() < shuffle_ratio:
                inputs[idx : idx + neighbor_span + 1] = (
                    inputs[idx + neighbor_span : idx - 1 : -1]
                    if idx > 0
                    else inputs[idx + neighbor_span :: -1]
                )

        return inputs

    def _determine_detect_types(self, coords):
        """
        Determine detection annotation type.
        """
        detect_types = []
        for coord in coords:
            assert coord.shape[1] == 2
            detect_type = self._num_pts_to_detect_type(coord.shape[0])
            detect_types.append(detect_type)

        return detect_types

    def _convert_random_detect_types(
        self, coords, detect_types, annot_change_prob=0.15
    ):
        """
        Convert quad, polygon to single, box randomly.
        """

        for i, coord in enumerate(coords):
            if np.random.random() <= annot_change_prob:
                if np.random.random() <= 0.5:
                    # box
                    detect_types[i] = self.box_format.detect_type
                    coords[i] = np.reshape(
                        np.concatenate(
                            (np.min(coord, axis=0), np.max(coord, axis=0)),
                            0,
                        ),
                        (2, 2),
                    )
                else:
                    # single
                    detect_types[i] = self.single_format.detect_type
                    coords[i] = poly_center(coord)

        return coords, detect_types

    def _convert_only_single_type(self, coords, detect_types):
        """
        Convert all annotations to single.
        """
        for i, coord in enumerate(coords):
            coords[i] = poly_center(coord)
            detect_types[i] = self.single_format.detect_type

        return coords, detect_types

    def _convert_only_box_type(self, coords, detect_types):
        """
        Convert all annotations to box.
        """
        for i, coord in enumerate(coords):
            if coord.shape[0] <= 2:
                # In the case of single, it is impossible to convert to box.
                continue
            tlx, tly = np.min(coord, axis=0)
            brx, bry = np.max(coord, axis=0)
            coords[i] = np.array([[tlx, tly], [brx, bry]], dtype=np.float32)

            detect_types[i] = self.box_format.detect_type

        return coords, detect_types

    def _determine_last_detect_type(self, inputs):
        """
        Determine detect token for a last (fake) object.
        """
        nums = []
        for detect_token_type in [0, 1, 2, 3]:
            num = inputs.count(self._determine_detect_token(detect_token_type))
            nums.append((num, detect_token_type))

        return max(nums)[1]

    def _num_pts_to_detect_type(self, num_pts):
        """
        Determine detection annotation type.
        """
        assert num_pts in [
            detect_format.num_pts for detect_format in self.detect_formats
        ]

        for detect_format in self.detect_formats:
            if num_pts == detect_format.num_pts:
                return detect_format.detect_type

    def _detect_type_to_num_pts(self, detect_type):
        """
        Determine the number of points by using detection annotation.
        """
        assert detect_type in [
            detect_format.detect_type for detect_format in self.detect_formats
        ]

        for detect_format in self.detect_formats:
            if detect_type == detect_format.detect_type:
                return detect_format.num_pts

    def _determine_detect_token(self, detect_type):
        """
        Determine token idx of detection annotation type.
        """
        assert detect_type in [
            detect_format.detect_type for detect_format in self.detect_formats
        ]
        for detect_format in self.detect_formats:
            if detect_type == detect_format.detect_type:
                return self.tokenizer.vocab[detect_format.token]

    def _make_object_sequence(
        self,
        category,
        coords,
        width,
        height,
        text=None,
        detect_type=0,
        input_perturb_prob=0.2,
    ):
        """
        Compute sequence corresponding to the object.
        """
        if not self.w_noise_augmentation:
            assert category != 0

        bin_size = self.tokenizer.bin_size
        inputs, targets = [], []
        tasks = []

        detect_token = self._determine_detect_token(detect_type)
        inputs.append(detect_token)
        targets.append(self.ignore_idx)
        tasks.append(self.detect_task_id)

        for k in range(coords.shape[0]):
            x, y = coords[k, :]

            x_bin = np.floor(x / width * (bin_size - 1)).astype(np.int32)
            y_bin = np.floor(y / height * (bin_size - 1)).astype(np.int32)

            # x_idx = self.tokenizer.encode_coord(x_bin)
            # y_idx = self.tokenizer.encode_coord(y_bin)
            x_idx, y_idx = self.tokenizer.encode_coord_xy(x_bin, y_bin)

            if (
                self.input_perturbation
                and self.coord_out_token not in [x_idx, y_idx]
                and np.random.random() < input_perturb_prob
            ):
                input_x_idx, input_y_idx = self._apply_coord_perturbation(
                    x_bin, y_bin, bin_size
                )
            else:
                input_x_idx, input_y_idx = x_idx, y_idx

            if self.coord_order == "xy":
                inputs.extend([input_x_idx, input_y_idx])
            else:
                inputs.extend([input_y_idx, input_x_idx])

            if category == 1:
                # text
                target_x_idx, target_y_idx = x_idx, y_idx
            elif category == -1 or self.eos_only_end or k > 0:
                # DC or noise's the other pts
                target_x_idx, target_y_idx = self.ignore_idx, self.ignore_idx
            else:
                # noise's first pts
                target_x_idx, target_y_idx = self.tokenizer.eos, self.ignore_idx

            if self.coord_order == "xy":
                targets.extend([target_x_idx, target_y_idx])
            else:
                targets.extend([target_y_idx, target_x_idx])

            tasks.extend([0, 0])

        input_tokens, target_tokens = self._make_transcriptions_tokens(
            self.drop_token_prob, category, text, self.max_text_length
        )

        inputs.extend(input_tokens)
        targets.extend(target_tokens)
        tasks.extend([self.recog_task_id] * len(target_tokens))

        return inputs, targets, tasks

    def _apply_coord_perturbation(
        self, x_bin, y_bin, bin_size, input_perturb_span_ratio=0.005
    ):
        """
        Coordinate perturbation.
        """
        input_perturb_span = int(bin_size * input_perturb_span_ratio)
        perturb_x_bin = np.random.randint(-input_perturb_span, input_perturb_span + 1)
        perturb_y_bin = np.random.randint(-input_perturb_span, input_perturb_span + 1)

        perturb_x_bin = np.clip(x_bin + perturb_x_bin, 0, bin_size - 1)
        perturb_y_bin = np.clip(y_bin + perturb_y_bin, 0, bin_size - 1)
        return self.tokenizer.encode_coord_xy(perturb_x_bin, perturb_y_bin)

    def _make_transcriptions_tokens(
        self, drop_token_prob, category, text, max_text_length
    ):
        """
        Determine tokens for recognition transcription.
        """
        text = text[:max_text_length]
        is_input_mask = False
        if category == -1 or np.random.random() < drop_token_prob:
            input_tokens = [self.mask_token] + (
                [self.tokenizer.pad] * (max_text_length - 1)
                if self.fixed_text_len
                else [self.tokenizer.vocab["[text_eos]"]]
            )
            is_input_mask = True
        else:
            input_tokens = text + (
                [self.tokenizer.pad] * (max_text_length - len(text))
                if self.fixed_text_len
                else [self.tokenizer.vocab["[text_eos]"]]
            )

        if category == -1 or is_input_mask:
            # DC
            target_tokens = [self.ignore_idx] * len(input_tokens)
        elif category == 1:
            # text
            target_tokens = text + (
                [self.tokenizer.pad] * (max_text_length - len(text))
                if self.fixed_text_len
                else [self.tokenizer.vocab["[text_eos]"]]
            )
        else:
            # noise
            target_tokens = [self.noise_token] + (
                [self.tokenizer.pad] * (max_text_length - 1)
                if self.fixed_text_len
                else [self.tokenizer.vocab["[text_eos]"]]
            )

        return input_tokens, target_tokens

    def _add_roi_point_prompt(self, inputs, targets, tasks, roi_bin):
        """
        Add roi prompt in sequence.
        """
        assert len(roi_bin) == 4
        roi_prompt = [self.roi_token] + [
            self.tokenizer.encode_coord(coord_bin) for coord_bin in roi_bin
        ]

        if self.coord_order == "yx":
            roi_prompt[1], roi_prompt[2] = roi_prompt[2], roi_prompt[1]
            roi_prompt[3], roi_prompt[4] = roi_prompt[4], roi_prompt[3]

        inputs = roi_prompt + inputs
        targets = [self.ignore_idx] * len(roi_prompt) + targets
        tasks = [self.detect_task_id] * len(roi_prompt) + tasks

        return inputs, targets, tasks

    def _add_order_prompt(self, inputs, targets, tasks, order_span):
        """
        Add order prompt in sequence.
        """
        assert len(order_span) == 2
        order_start_idx, order_end_idx = order_span
        order_start_idx = self.tokenizer.encode_order(order_start_idx)
        order_end_idx = self.tokenizer.encode_order(order_end_idx)

        inputs = [
            self.order_token,
            order_start_idx,
            order_end_idx,
        ] + inputs
        targets = [self.ignore_idx] * 3 + targets
        tasks = [self.detect_task_id] * 3 + tasks

        return inputs, targets, tasks

    def _add_point_prompt(self, inputs, targets, tasks, point_bin):
        """
        Add point prompt in sequence.
        """
        assert len(point_bin) == 2
        point_prompt = [self.point_token] + [
            self.tokenizer.encode_coord(coord_bin) for coord_bin in point_bin
        ]

        if self.coord_order == "yx":
            point_prompt[1], point_prompt[2] = point_prompt[2], point_prompt[1]

        inputs = point_prompt + inputs
        targets = [self.ignore_idx] * len(point_prompt) + targets
        tasks = [self.detect_task_id] * len(point_prompt) + tasks

        return inputs, targets, tasks

    def _make_sequence_for_ocr(
        self,
        candid_texts,
        candid_coords,
        candid_detect_types,
        candid_dcs,
        img_h,
        img_w,
        prompt_input=None,
        prob_duplicate_noise=0.5,
        prob_shift_noise=0.5,
    ):
        """
        Make input and target sequence.
        Args:
            prob_duplicate_noise: duplicated noise ratio out of the whole noise (duplicated noise + random noise)
            prob_shift_noise: center shifted noise ratio out of the random noise (center shifted noise + randomly generated noise)
        """
        selected_texts = []
        selected_coords = []
        selected_categories = []
        selected_detect_types = []

        if self.prompt == "roi":
            bin_size = self.tokenizer.bin_size
            tlx_bin, tly_bin, brx_bin, bry_bin = prompt_input
            tlx = tlx_bin / (bin_size - 1) * img_w
            tly = tly_bin / (bin_size - 1) * img_h
            brx = brx_bin / (bin_size - 1) * img_w
            bry = bry_bin / (bin_size - 1) * img_h
            roi = tlx, tly, brx, bry
        else:
            roi = None

        if self.n_object <= len(candid_texts):
            ids = list(range(self.n_object))
        else:
            ids = list(range(len(candid_texts)))
            if self.w_noise_augmentation:
                ids = ids + [-1] * (self.n_object - len(candid_texts))

        for id in ids:
            if id == -1:
                if len(candid_coords) > 0:
                    if np.random.random() <= prob_duplicate_noise:
                        noise_coord, noise_coord_id = make_duplicate_coord(
                            candid_coords, img_w, img_h, roi
                        )
                        noise_detect_type = candid_detect_types[noise_coord_id]
                    else:
                        if np.random.random() <= prob_shift_noise:
                            noise_coord, noise_coord_id = make_shift_coord(
                                candid_coords, img_w, img_h, roi
                            )
                            noise_detect_type = candid_detect_types[noise_coord_id]
                        else:
                            noise_detect_type = random.randint(
                                0, len(self.detect_formats) - 1
                            )
                            noise_num_pts = self._detect_type_to_num_pts(
                                noise_detect_type
                            )
                            noise_coord = make_random_coord(
                                img_w, img_h, noise_num_pts, roi
                            )
                else:
                    noise_detect_type = random.randint(0, 3)
                    noise_num_pts = self._detect_type_to_num_pts(noise_detect_type)
                    noise_coord = make_random_coord(img_w, img_h, noise_num_pts, roi)

                noise_text = make_random_tokens(
                    self.max_text_length, self.tokenizer.char_vocab_range
                )

                category = 0  # noise
                selected_texts.append(noise_text)
                selected_coords.append(noise_coord)
                selected_categories.append(category)
                selected_detect_types.append(noise_detect_type)
            else:
                category = -1 if candid_dcs[id] else 1  # valid text or dc
                selected_texts.append(candid_texts[id])
                selected_coords.append(candid_coords[id])
                selected_categories.append(category)
                selected_detect_types.append(candid_detect_types[id])

        return (
            selected_texts,
            selected_coords,
            selected_categories,
            selected_detect_types,
        )

    def _sampling_roi(
        self,
        coords,
        texts,
        detect_types,
        img_w,
        img_h,
        prob_at_least_one=0.8,
        coord_bin_margin=0.005,
        max_tries=50,
    ):
        """
        Generate RoI and refine instances.
        """
        bin_size = self.tokenizer.bin_size
        min_n_object = 1 if np.random.random() <= prob_at_least_one else 0
        coord_bin_margin = int(bin_size * coord_bin_margin)

        for _ in range(max_tries):
            roi_tlx = np.random.randint(bin_size)
            roi_tly = np.random.randint(bin_size)
            roi_brx = np.random.randint(roi_tlx + 1, bin_size)
            roi_bry = np.random.randint(roi_tly + 1, bin_size)

            refined_coords, refined_texts, refined_detect_types = [], [], []

            for coord, text, detect_type in zip(coords, texts, detect_types):
                instance_x, instance_y = poly_center(coord)[0].tolist()

                instance_x = np.floor(instance_x / img_w * (bin_size - 1)).astype(
                    np.int32
                )
                instance_y = np.floor(instance_y / img_h * (bin_size - 1)).astype(
                    np.int32
                )
                if (
                    instance_x >= roi_tlx - coord_bin_margin
                    and instance_x <= roi_brx + coord_bin_margin
                    and instance_y >= roi_tly - coord_bin_margin
                    and instance_y <= roi_bry + coord_bin_margin
                ):
                    refined_coords.append(coord)
                    refined_texts.append(text)
                    refined_detect_types.append(detect_type)

            n_valid_object = len([text for text in refined_texts if text != ""])
            if n_valid_object >= min_n_object:
                return (
                    [roi_tlx, roi_tly, roi_brx, roi_bry],
                    refined_coords,
                    refined_texts,
                    refined_detect_types,
                    True,
                )

        return (
            [0, 0, bin_size - 1, bin_size - 1],
            coords,
            texts,
            detect_types,
            False,
        )

    def _sampling_orders(
        self,
        coords,
        texts,
        detect_types,
        max_order,
        dcs_inputs=False,
        max_n_object=100,
        zero_start_prob=0.2,
        max_end_idx_prob=0.5,
        prob_at_least_one=0.8,
        span_margin=5,
    ):
        """
        Generate start/end idx and refine instances.
        """
        valid_coords, valid_texts, valid_detect_types = [], [], []
        for coord, text, detect_type in zip(coords, texts, detect_types):
            if not dcs_inputs and text == "":
                continue
            valid_coords.append(coord)
            valid_texts.append(text)
            valid_detect_types.append(detect_type)

        if np.random.random() <= zero_start_prob:
            start_idx = 0
        else:
            if np.random.random() <= prob_at_least_one:
                start_idx = (
                    min(max_order, np.random.randint(len(valid_coords)))
                    if len(valid_coords)
                    else 0
                )
            else:
                start_idx = np.random.randint(
                    min(max_order, len(valid_coords) + span_margin)
                )

        if np.random.random() <= max_end_idx_prob:
            end_idx = start_idx + max_n_object - 1
        else:
            end_idx = start_idx + np.random.randint(max_n_object)

        start_idx = min(start_idx, max_order - 1)
        end_idx = min(end_idx, max_order - 1)

        sampled_coords, sampled_texts, sampled_detect_types = [], [], []

        for i in range(start_idx, min(end_idx + 1, len(valid_coords))):
            sampled_coords.append(valid_coords[i])
            sampled_texts.append(valid_texts[i])
            sampled_detect_types.append(valid_detect_types[i])

        return [start_idx, end_idx], sampled_coords, sampled_texts, sampled_detect_types

    def _sampling_starting_point(
        self,
        coords,
        texts,
        detect_types,
        img_w,
        img_h,
        prob_at_start=0.5,
        prob_at_random=0.5,
        prob_at_least_one=0.8,
        coord_bin_margin=0.005,
        max_tries=50,
    ):
        """
        Generate point and refine instances.
        """
        bin_size = self.tokenizer.bin_size
        min_n_object = 1 if np.random.random() <= prob_at_least_one else 0
        coord_bin_margin = int(bin_size * coord_bin_margin)

        for _ in range(max_tries):
            if np.random.random() <= prob_at_start:
                start_x, start_y = 0, 0
            else:
                if len(coords) == 0 or np.random.random() <= prob_at_random:
                    start_x = np.random.randint(bin_size)
                    start_y = np.random.randint(bin_size)
                else:
                    sampled_idx = np.random.randint(len(coords))
                    start_x, start_y = np.mean(coords[sampled_idx], axis=0)
                    start_x = np.floor(start_x / img_w * (bin_size - 1)).astype(
                        np.int32
                    )
                    start_y = np.floor(start_y / img_h * (bin_size - 1)).astype(
                        np.int32
                    )

            refined_coords, refined_texts, refined_detect_types = [], [], []

            for coord, text, detect_type in zip(coords, texts, detect_types):
                instance_x, instance_y = poly_center(coord)[0].tolist()

                instance_x = np.floor(instance_x / img_w * (bin_size - 1)).astype(
                    np.int32
                )
                instance_y = np.floor(instance_y / img_h * (bin_size - 1)).astype(
                    np.int32
                )
                if instance_y > start_y or (
                    start_y - instance_y <= coord_bin_margin
                    and start_x - instance_x <= coord_bin_margin
                ):
                    refined_coords.append(coord)
                    refined_texts.append(text)
                    refined_detect_types.append(detect_type)

            n_valid_object = len([text for text in refined_texts if text != ""])
            if n_valid_object >= min_n_object:
                return (
                    [start_x, start_y],
                    refined_coords,
                    refined_texts,
                    refined_detect_types,
                    True,
                )

        return (
            [0, 0],
            coords,
            texts,
            detect_types,
            False,
        )
