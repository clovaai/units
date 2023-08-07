"""
UNITS
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import numpy as np
import torch
import torch.nn as nn
from pydantic import StrictBool, StrictInt


@torch.no_grad()
def greedy_decode(
    batch_size,
    device,
    prev_outputs,
    n_vocab,
    max_length,
    pix2seq_embed,
    pix2seq_pos,
    pix2seq_decoder,
    pix2seq_head,
    go=1,
    eos=2,
    noise=5,
    text_eos=None,
    prompt=None,
    ignore_eos_noise=False,
    detect_type=None,
    text_length=25,
    fixed_text_len=True,
    multiple_experts=False,
    **decoder_kwargs,
):
    """
    Args:
        batch_size (int): batch size for decoding
        device (Union[str, torch.device]): device for decoding
        n_vocab (int): number of vocabulary for decoding
        max_length (int): maximum length of sequences that decoded
        pix2seq_embed (Callable): function that returns tensor given inputs
        pix2seq_pos (Tensor[float]): positional encoding (max_length, dim)
        pix2seq_decoder (Callable): transformer decoder
        pix2seq_head (Callable): function that returns vocab id given decoder output
        **decoder_kwargs: Rest of the input that should be given to the text decoder

    Returns:
        logits (Tensor[float]): logit before softmax (batch, max_length, n_vocab)
        tokens (Tensor[int64]): decoded token ids (batch, max_length)
    """
    start_token = prompt[0] if len(prompt) else go
    dec = torch.zeros(batch_size, max_length, n_vocab, device=device)
    out_texts = torch.ones(batch_size, max_length, dtype=torch.int64, device=device)
    texts = torch.ones(batch_size, 1, dtype=torch.int64, device=device) * start_token
    cache = None
    eos_flag = torch.full((batch_size,), False, dtype=torch.bool, device=device)
    expert_idx = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)

    detect_type, detect_type_token, num_pts = detect_type
    span = 1 + num_pts * 2 + text_length
    next_recog_start_idx = [-1] * batch_size

    for i in range(max_length):
        # Enforce the beginning of the text to be the previous output.
        if prev_outputs is not None and i < len(prev_outputs[0]):
            for batch_i in range(len(prev_outputs)):
                texts[batch_i, 0] = prev_outputs[batch_i][i]

        embed = pix2seq_embed(texts)

        pos = pix2seq_pos[i : i + 1].unsqueeze(0).expand(batch_size, -1, -1)

        if multiple_experts:
            expert_idx = refine_expert_idx(i, next_recog_start_idx, expert_idx)

        decode_res = pix2seq_decoder(
            embed,
            pos,
            memory=cache,
            expert_idx=expert_idx if multiple_experts else None,
            **decoder_kwargs,
        )
        decode = decode_res[0]
        cache = decode_res[-1]

        output_class = pix2seq_head(decode[-1])
        next_step = output_class[:, -1]

        if ignore_eos_noise:
            next_step[:, eos] = -torch.tensor(float("inf"))
            next_step[:, noise] = -torch.tensor(float("inf"))

        next_token = next_step.argmax(1)

        # Enforce the beginning of the text to be the prompt
        if i < len(prompt):
            if i == len(prompt) - 1:
                next_token = (
                    torch.ones(batch_size, dtype=torch.int64, device=device) * go
                )
            else:
                next_token = (
                    torch.ones(batch_size, dtype=torch.int64, device=device)
                    * prompt[i + 1]
                )

        if fixed_text_len and (i - len(prompt)) % span == 0:
            next_token = (
                torch.ones(batch_size, dtype=torch.int64, device=device)
                * detect_type_token
            )
            if multiple_experts:
                expert_idx = torch.zeros(
                    batch_size, 1, dtype=torch.int64, device=device
                )
                next_recog_start_idx = [i + num_pts * 2 + 1] * batch_size
        elif fixed_text_len is False and i > 0:
            for j in range(texts.shape[0]):
                if texts[j, 0] in [text_eos, go]:
                    next_token[j] = detect_type_token
                    if multiple_experts:
                        expert_idx[j] = 0
                        next_recog_start_idx[j] = i + num_pts * 2 + 1

        texts = next_token.unsqueeze(1)
        out_texts[:, i] = next_token
        dec[:, i, :] = next_step
        eos_cond = next_token == eos
        eos_flag = eos_flag | eos_cond

        if eos_flag.all():
            break

    return torch.softmax(dec, dim=-1), out_texts


def refine_expert_idx(curr_step_i, next_recog_start_idx, expert_idx):
    for batch_i, step_i in enumerate(next_recog_start_idx):
        if step_i < 0:
            continue
        if curr_step_i == step_i:
            expert_idx[batch_i] = 1

    return expert_idx


class UnitsDecoder(nn.Module):
    """
    Units Decoder
    Args:
        dim (int): hidden dimension
        decoder_length (int): decoder max length
    """

    def __init__(
        self,
        dim: StrictInt,
        max_text_length: StrictInt,
        pix2seq_dec: nn.Module,
        loss_criterion: nn.Module,
        n_object: StrictInt,
        decoder_length: StrictInt,
        tokenizer,
        prompt,
        detect_type,
        fixed_text_len: StrictBool,
        coord_order,
        iterative_decoding=False,
        max_iter=15,
        n_overlap=4,
    ):
        super().__init__()

        self.dim = dim
        self.max_text_length = max_text_length

        self.pix2seq_decoder = pix2seq_dec

        self.tokenizer = tokenizer
        self.n_vocab = self.tokenizer.n_vocab

        self.go = tokenizer.go
        self.eos = tokenizer.eos
        # special tokens index (noise, special prompts)
        self.noise = tokenizer.vocab["[noise]"]
        self.roi = tokenizer.vocab["[roi]"]
        self.order = tokenizer.vocab["[order]"]
        self.point = tokenizer.vocab["[point]"]
        self.text_eos = tokenizer.vocab["[text_eos]"]

        self.bin_size = tokenizer.bin_size
        self.coord_vocab_range = (
            tokenizer.encode_coord(0),
            tokenizer.encode_coord(self.bin_size - 1),
        )

        self.prompt = prompt
        if prompt is not None:
            assert prompt in ["roi", "order", "point"]

        self.decoder_length = decoder_length
        if self.prompt == "order":
            self.order_vocab_range = (
                tokenizer.encode_order(0),
                tokenizer.encode_order(n_object - 1),
            )

        self.pix2seq_head = nn.Linear(dim, self.n_vocab)
        self.pix2seq_embed = nn.Embedding(self.n_vocab, dim, padding_idx=None)
        self.pix2seq_pos = nn.Parameter(torch.randn(decoder_length, dim) * 0.02)

        self.loss_criterion = loss_criterion

        num_pts_dict = {
            "single": 1,
            "box": 2,
            "quad": 4,
            "polygon": 16,
        }
        assert detect_type in num_pts_dict
        self.num_pts_dict = num_pts_dict
        self.detect_type = (
            f"[{detect_type}]",
            self.tokenizer.vocab[f"[{detect_type}]"],
            num_pts_dict[detect_type],
        )
        self.token2npts = {
            self.tokenizer.vocab[f"[{dtype}]"]: num_pts
            for dtype, num_pts in num_pts_dict.items()
        }

        self.fixed_text_len = fixed_text_len
        self.coord_order = coord_order
        assert coord_order in ["xy", "yx"]

        self.iterative_decoding = iterative_decoding
        self.max_iter = max_iter if self.iterative_decoding else 1
        self.n_overlap = n_overlap

    def forward(
        self,
        batch,
        source_feat,
        mask,
        detect_type=None,
        threshold=0.0,
    ):
        """
        Args:
            batch (Batch): instance of Batch with Pix2SeqBatch fields
            source_feat (Tensor[float]): flattened and concatenated features (N, ?, dim)
            mask (Tensor[bool]): True for non-image areas and False for image areas (N, H, W)
            threshold (float): confidence threshold for filter out entities
        """

        if self.training:
            return self.forward_train(
                batch,
                source_feat,
                mask,
            )

        else:
            return self.forward_eval(
                batch,
                source_feat,
                mask,
                detect_type,
                threshold,
            )

    def forward_train(
        self,
        batch,
        feats,
        mask,
    ):
        outputs = {}
        pix2seq_in = batch.units.units_inputs
        pix2seq_embed = self.pix2seq_embed(pix2seq_in)
        pix2seq_pos = self.pix2seq_pos
        pix2seq_pos = pix2seq_pos.unsqueeze(0).expand(pix2seq_in.shape[0], -1, -1)

        source_mask = torch.unsqueeze(mask, 1).expand(-1, pix2seq_pos.shape[1], -1)
        source_mask = source_mask.unsqueeze(1)
        # source_mask = None

        expert_idx = batch.units.tasks
        pix2seq_out, _ = self.pix2seq_decoder(
            pix2seq_embed, pix2seq_pos, feats, source_mask, expert_idx=expert_idx
        )

        output_logit = self.pix2seq_head(pix2seq_out[-1])

        loss = self.loss_criterion(
            output_logit.reshape(-1, output_logit.shape[-1]).contiguous(),
            batch.units.units_targets.reshape(-1),
        )

        outputs.update(
            {
                "total_loss": loss,
            }
        )

        return outputs

    def forward_eval(
        self,
        batch,
        feats,
        mask,
        detect_type,
        threshold,
    ):
        if detect_type is not None:
            assert detect_type in self.num_pts_dict
            self.num_pts_dict = self.num_pts_dict
            self.detect_type = (
                f"[{detect_type}]",
                self.tokenizer.vocab[f"[{detect_type}]"],
                self.num_pts_dict[detect_type],
            )
            self.token2npts = {
                self.tokenizer.vocab[f"[{dtype}]"]: num_pts
                for dtype, num_pts in self.num_pts_dict.items()
            }

        outputs = {}

        if self.prompt == "roi":
            prompt_input = [
                self.roi,
                self.coord_vocab_range[0],
                self.coord_vocab_range[0],
                self.coord_vocab_range[1],
                self.coord_vocab_range[1],
            ]
        elif self.prompt == "order":
            prompt_input = [
                self.order,
                self.order_vocab_range[0],
                self.order_vocab_range[1],
            ]
        elif self.prompt == "point":
            prompt_input = [
                self.point,
                self.coord_vocab_range[0],
                self.coord_vocab_range[0],
            ]
        else:
            prompt_input = []

        source_mask = torch.unsqueeze(torch.unsqueeze(mask, 1), 2)
        # source_mask = None

        prev_outputs = None

        merged_output_logit, merged_output_ids = [], []
        for _ in range(self.max_iter):
            output_logit, output_ids = greedy_decode(
                batch.images.shape[0],
                feats.device,
                prev_outputs,
                self.n_vocab,
                self.decoder_length,
                self.pix2seq_embed,
                self.pix2seq_pos,
                self.pix2seq_decoder,
                self.pix2seq_head,
                go=self.go,
                eos=self.eos,
                noise=self.noise,
                text_eos=self.text_eos,
                prompt=prompt_input,
                source=feats,
                source_mask=source_mask,
                detect_type=self.detect_type,
                text_length=self.max_text_length,
                fixed_text_len=self.fixed_text_len,
                multiple_experts=self.pix2seq_decoder.layers[0].n_experts > 1,
            )

            merged_output_logit.append(output_logit)
            merged_output_ids.append(output_ids)
            if self.iterative_decoding:
                prev_outputs = self.determine_last_objects(
                    output_ids, detect_type=self.detect_type, n_overlap=self.n_overlap
                )
                prev_outputs = self.add_point_prompt(
                    prev_outputs, detect_type=self.detect_type
                )

        vertices = []
        scores = []
        texts = []

        eos_flags = [False] * output_ids.shape[0]

        for i in range(len(merged_output_logit)):
            output_ids = merged_output_ids[i]
            output_logit = merged_output_logit[i]

            for batch_i in range(output_ids.shape[0]):
                if eos_flags[batch_i] == True:
                    continue

                if self.eos in output_ids[batch_i, len(prompt_input) :].tolist():
                    eos_flags[batch_i] = True

                h, w = batch.samples[batch_i].image_size

                curr_vertices, curr_scores, curr_texts = self.convert_seq2ocr(
                    output_logit[batch_i, len(prompt_input) :].tolist(),
                    output_ids[batch_i, len(prompt_input) :].tolist(),
                    h,
                    w,
                    threshold,
                    self.fixed_text_len,
                    self.text_eos,
                    self.coord_order,
                )
                if i == 0:
                    vertices.append(curr_vertices)
                    scores.append(curr_scores)
                    texts.append(curr_texts)
                else:
                    if (
                        curr_vertices is not None
                        and len(curr_vertices) > self.n_overlap
                    ):
                        vertices[batch_i].extend(curr_vertices[self.n_overlap :])
                        scores[batch_i].extend(curr_scores[self.n_overlap :])
                        texts[batch_i].extend(curr_texts[self.n_overlap :])

        outputs.update(
            {
                "texts": texts,
                "vertices": vertices,
                "scores": scores,
            }
        )

        return outputs

    def convert_seq2ocr(
        self,
        logit,
        ids,
        h,
        w,
        threshold=0.0,
        fixed_text_len=True,
        text_eos=0,
        coord_order="xy",
    ):
        """
        Convert sequence to OCR results.
        """
        if self.tokenizer.eos in ids:
            first_eos_ids = ids.index(self.tokenizer.eos)
            ids = ids[:first_eos_ids]
            logit = logit[:first_eos_ids]

        min_idx_coord_vocab, max_idx_coord_vocab = self.coord_vocab_range

        curr_vertices = []
        curr_scores = []
        curr_texts = []

        # Split instances by detect type tokens.
        start_ids, end_ids, num_pts_list = [], [], []
        for i, idx in enumerate(ids):
            if idx in self.token2npts:
                start_ids.append(i)

                num_pts = self.token2npts[idx]
                num_pts_list.append(num_pts)

                if fixed_text_len:
                    span = 2 * num_pts + 1 + self.max_text_length
                    if i + span - 1 < self.decoder_length:
                        end_ids.append(i + span - 1)
                elif idx == text_eos:
                    end_ids.append(i)

        # Convert tokens to vertices & text transcriptions.
        for start_id, end_id, num_pts in zip(start_ids, end_ids, num_pts_list):
            object = ids[start_id : end_id + 1]
            if any(
                [
                    p < min_idx_coord_vocab or p > max_idx_coord_vocab
                    for p in object[1 : 1 + 2 * num_pts]
                ]
            ):
                continue

            if len(object[1 : 1 + 2 * num_pts]) < 2 * num_pts:
                continue

            score = logit[start_id : end_id + 1]

            mean_score = []
            for i, p in enumerate(object[1 + 2 * num_pts :]):
                if p == 0:
                    break
                mean_score.append(score[i + 1 + 2 * num_pts][p])
            object_score = np.mean(mean_score)

            vertices = list(
                map(
                    lambda x: self.tokenizer.decode_coord(x),
                    object[1 : 1 + 2 * num_pts],
                )
            )

            vertices = np.array(vertices, dtype=np.float32).reshape(num_pts, 2)
            if coord_order == "xy":
                vertices[:, 0], vertices[:, 1] = vertices[:, 0] * w / (
                    self.bin_size - 1
                ), vertices[:, 1] * h / (self.bin_size - 1)
            else:
                vertices[:, 0], vertices[:, 1] = vertices[:, 1] * h / (
                    self.bin_size - 1
                ), vertices[:, 0] * w / (self.bin_size - 1)

            text = object[1 + 2 * num_pts :]

            if object_score >= threshold:
                curr_vertices.append(vertices)
                curr_scores.append(object_score)
                curr_texts.append(text)

        if curr_vertices == []:
            curr_vertices = None
            curr_scores = None
            curr_texts = None

        return curr_vertices, curr_scores, curr_texts

    def determine_last_objects(
        self, output_ids, detect_type, fixed_text_len=True, n_overlap=1
    ):
        """
        Determine last object tokens (n_overlap) for iterative decoding.
        """
        detect_type, detect_type_token, num_pts = detect_type

        # TODO: implement fixed_text_len=False
        assert fixed_text_len == True
        span = 2 * num_pts + self.max_text_length

        last_object_tokens = []
        for batch_i in range(output_ids.shape[0]):
            batch_output_ids = output_ids[batch_i].tolist()
            batch_last_object_tokens = []
            if self.tokenizer.eos in batch_output_ids:
                batch_last_object_tokens.append(detect_type_token)
                batch_last_object_tokens.extend([self.tokenizer.eos] * span)

                for _ in range(n_overlap - 1):
                    batch_last_object_tokens.extend([self.tokenizer.eos])
                    batch_last_object_tokens.extend([self.tokenizer.eos] * span)
            else:
                detect_type_ids = []
                for token_i, token in enumerate(batch_output_ids):
                    if token == detect_type_token:
                        detect_type_ids.append(token_i)

                # If last object is full
                if detect_type_ids[-1] < self.decoder_length - span:
                    for j in range(n_overlap - 1, -1, -1):
                        batch_last_object_tokens.extend(
                            batch_output_ids[
                                detect_type_ids[-1 - j] : detect_type_ids[-1 - j]
                                + span
                                + 1
                            ]
                        )
                else:
                    for j in range(n_overlap - 1, -1, -1):
                        batch_last_object_tokens.extend(
                            batch_output_ids[
                                detect_type_ids[-2 - j] : detect_type_ids[-2 - j]
                                + span
                                + 1
                            ]
                        )
            last_object_tokens.append(batch_last_object_tokens)

        return last_object_tokens

    def add_point_prompt(self, prev_inputs, detect_type):
        """
        Add starting-point prompt tokens to the beginning of the input tokens.
        """
        detect_type, detect_type_token, num_pts = detect_type

        for batch_i in range(len(prev_inputs)):
            coords = prev_inputs[batch_i][1 : 2 * num_pts + 1]
            if self.tokenizer.eos in coords:
                prev_inputs[batch_i] = [
                    self.point,
                    self.coord_vocab_range[0],
                    self.coord_vocab_range[0],
                    self.tokenizer.go,
                ] + prev_inputs[batch_i]
            else:
                coords = [self.tokenizer.decode_coord(coord) for coord in coords]
                prompt_x, prompt_y = self.tokenizer.encode_coord_xy(
                    int(np.mean(coords[::2])), int(np.mean(coords[1::2]))
                )
                prev_inputs[batch_i] = [
                    self.point,
                    prompt_x,
                    prompt_y,
                    self.tokenizer.go,
                ] + prev_inputs[batch_i]

        return prev_inputs
