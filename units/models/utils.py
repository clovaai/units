import random

import numpy as np
from shapely.geometry import LineString


def poly_center(poly_pts):
    """
    Calculate the center point of a polygon.
    """
    num_pts = poly_pts.shape[0]

    if num_pts % 2 == 0 and num_pts > 2:
        line1 = LineString(poly_pts[int(num_pts / 2) :])
        line2 = LineString(poly_pts[: int(num_pts / 2)])
        mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
        mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
        return np.expand_dims((mid_pt1 + mid_pt2) / 2, axis=0)
    else:
        single_x, single_y = np.mean(poly_pts, axis=0)
        return np.array([[single_x, single_y]], dtype=np.float32)


def make_duplicate_coord(
    coords, input_w, input_h, roi=None, min_range=0.0, max_range=0.1
):
    """
    Jitter the bbox, and top-left pts should be in RoI.
    """
    coord_id = np.random.randint(len(coords))
    num_pts = coords[coord_id].shape[0]
    perturb_val = np.random.normal(min_range, max_range / 2, coords[coord_id].shape)

    w, h = input_w, input_h
    if roi is None:
        # w, h = input_w, input_h
        min_tlx, max_tlx = 0, input_w - 1
        min_tly, max_tly = 0, input_h - 1
    else:
        roi_tlx, roi_tly, roi_brx, roi_bry = roi
        # w = roi_brx - roi_tlx + 1
        # h = roi_bry - roi_tly + 1
        min_tlx, max_tlx = roi_tlx, roi_brx
        min_tly, max_tly = roi_tly, roi_bry

    coord_w_perturb = coords[coord_id] + perturb_val * np.array(
        [[w, h] for _ in range(num_pts)]
    )

    min_x = np.array([min_tlx] + [0] * (num_pts - 1))
    max_x = np.array([max_tlx] + [input_w - 1] * (num_pts - 1))
    min_y = np.array([min_tly] + [0] * (num_pts - 1))
    max_y = np.array([max_tly] + [input_h - 1] * (num_pts - 1))

    return (
        np.stack(
            [
                np.clip(coord_w_perturb[:, 0], min_x, max_x),
                np.clip(coord_w_perturb[:, 1], min_y, max_y),
            ],
            1,
        ),
        coord_id,
    )


def make_shift_coord(coords, input_w, input_h, roi=None):
    """
    Shift bbox without changing the bbox height and width.
    """
    coord_id = np.random.randint(len(coords))

    coord_topleft = [
        coords[coord_id][0, 0],
        coords[coord_id][0, 1],
    ]

    # Sample new bbox centers randomly.
    if roi is None:
        min_x, max_x = 0, input_w - 1
        min_y, max_y = 0, input_h - 1
    else:
        roi_tlx, roi_tly, roi_brx, roi_bry = roi
        min_x, max_x = roi_tlx, roi_brx
        min_y, max_y = roi_tly, roi_bry

    new_coord_topleft = [
        np.random.randint(min_x, max_x + 1),
        np.random.randint(min_y, max_y + 1),
    ]

    shift = [
        new_coord_topleft[0] - coord_topleft[0],
        new_coord_topleft[1] - coord_topleft[1],
    ]

    return (
        np.stack(
            [
                np.clip(coords[coord_id][:, 0] + shift[0], 0, input_w - 1),
                np.clip(coords[coord_id][:, 1] + shift[1], 0, input_h - 1),
            ],
            1,
        ),
        coord_id,
    )


def make_random_coord(input_w, input_h, num_pts=4, roi=None):
    """
    Generate random bbox with max size specified within [0, img_wh], and top-left pts should be in RoI.
    """
    if roi:
        roi_tlx, roi_tly, roi_brx, roi_bry = roi
        min_tlx, max_tlx = roi_tlx, roi_brx
        min_tly, max_tly = roi_tly, roi_bry
        if np.random.random() >= 0.5:
            min_other_x, max_other_x = roi_tlx, roi_brx
            min_other_y, max_other_y = roi_tly, roi_bry
        else:
            # Except top-left pts, the other points dont have to be in RoI
            min_other_x, max_other_x = 0, input_w - 1
            min_other_y, max_other_y = 0, input_h - 1
    else:
        min_tlx, max_tlx = 0, input_w - 1
        min_tly, max_tly = 0, input_h - 1
        min_other_x, max_other_x = 0, input_w - 1
        min_other_y, max_other_y = 0, input_h - 1

    coord = [
        [
            np.random.randint(min_tlx, max_tlx + 1),
            np.random.randint(min_tly, max_tly + 1),
        ]
    ] + [
        [
            np.random.randint(min_other_x, max_other_x + 1),
            np.random.randint(min_other_y, max_other_y + 1),
        ]
        for _ in range(num_pts - 1)
    ]
    return np.array(coord)


def make_random_tokens(max_text_length, char_vocab_range):
    """
    Make random noise text.
    """
    noise_text_length = np.random.randint(1, max_text_length)
    char_start_idx, char_end_idx = char_vocab_range
    noise_text = random.choices(
        list(range(char_start_idx, char_end_idx + 1)), k=noise_text_length
    )
    return noise_text
