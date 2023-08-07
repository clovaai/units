# python script/make_lmdb.py --split train/val/test

import argparse
import multiprocessing
import os
import pickle
from functools import partial
from glob import glob

import lmdb
import numpy as np
import orjson
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

annotation_path = "train_datasets"
synthtext_vocab = list(
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)

# annotation, img_path pair list
train_annotation_files = {
    # mixed
    "hiertext": ["annotations/hiertext/train.jsonl", "images/hiertext/train"],
    "textocr": [
        "annotations/textocr/TextOCR_0.1_train.json",
        "images/textocr/train_images",
    ],
    # poly (ours)
    "textocr.poly": [
        "annotations/textocr/TextOCR_0.1_train_poly.json",
        "images/textocr/train_images",
    ],
    # poly (AdelaiDet)
    "synthtext150k.poly.part1": [
        "annotations/synthtext150k/syntext1/train_poly.json",
        "images/synthtext150k/syntext_word_eng/",
    ],
    "synthtext150k.poly.part2": [
        "annotations/synthtext150k/syntext2/train_poly.json",
        "images/synthtext150k/emcs_imgs/",
    ],
    "ctw1500.poly": [
        "annotations/ctw1500/train_poly.json",
        "images/ctw1500/ctwtrain_text_image",
    ],
    "totaltext.poly": [
        "annotations/totaltext/train_poly.json",
        "images/totaltext/train_images",
    ],
    # box
    "icdar13": ["annotations/icdar13/ICDAR2013_FOCUSED_train.json", "images/icdar13"],
    # quad
    "icdar15": [
        "annotations/icdar15/ICDAR2015_Incidental_train.json",
        "images/icdar15/train",
    ],
    "mlt19": ["annotations/mlt19/gt.json", "images/mlt19/"],
}

val_annotation_files = {
    "textocr": [
        "annotations/textocr/TextOCR_0.1_val.json",
        "images/textocr/train_images",
    ],
    "hiertext": ["annotations/hiertext/validation.jsonl", "images/hiertext/validation"],
}

test_annotation_files = {
    "icdar13": [
        "annotations/icdar13/ICDAR2013_FOCUSED_test.json",
        "images/icdar13/test",
    ],
    "icdar15": [
        "annotations/icdar15/ICDAR2015_Incidental_test.json",
        "images/icdar15/test",
    ],
    "totaltext.poly": [
        "annotations/totaltext/test_poly.json",
        "images/totaltext/test_images",
    ],
    "ctw1500.poly": [
        "annotations/ctw1500/test_poly.json",
        "images/ctw1500/ctwtest_text_image",
    ],
}

annotation_files = {
    "train": train_annotation_files,
    "val": val_annotation_files,
    "test": test_annotation_files,
}


def read_json(fname):
    with open(fname) as f:
        return orjson.loads(f.read())


def to_record(row, dataset):
    image_id = row["image_id"]
    if dataset != "hiertext":
        filename = row["filename"]

    dc, words = [], []
    if dataset == "hiertext":
        word_id = 0
        for paragraph_id, paragraph in enumerate(row["paragraphs"]):
            for line_id, line in enumerate(paragraph["lines"]):
                for word in line["words"]:
                    vertices = word["vertices"]
                    text = word["text"]
                    legible = word["legible"]
                    # handwritten = word["handwritten"]
                    horizontal = not word["vertical"]
                    if text == "":
                        legible = False
                    if legible is False:
                        dc.append(np.array(vertices, dtype=np.float32))
                    else:
                        words.append(
                            (vertices, text, horizontal, word_id, line_id, paragraph_id)
                        )
                        word_id += 1
        filename = str(image_id) + ".jpg"
    elif dataset in [
        "icdar13",
        "icdar15",
        "mlt19",
        "textocr",
        "textocr.poly",
        "synthtext150k.poly.part1",
        "synthtext150k.poly.part2",
        "totaltext.poly",
        "ctw1500.poly",
    ]:
        for word_id, word in enumerate(row["words"]):
            vertices = [
                (x, y) for x, y in zip(word["vertices"][::2], word["vertices"][1::2])
            ]
            text = word["text"]
            legible = word["legible"]
            paragraph_id = None
            line_id = None
            horizontal = None
            if legible is False:
                dc.append(np.array(vertices, dtype=np.float32))
            else:
                words.append(
                    (vertices, text, horizontal, word_id, line_id, paragraph_id)
                )

    width = row["image_width"]
    height = row["image_height"]

    result = {
        "id": image_id,
        "words": words,
        "dcs": dc,
        "filename": filename,
        "orig_size": (width, height),
        "dataset_name": dataset,
    }
    return result


def worker(data, root, target, name, split, max_size):
    id, row = data
    record = to_record(row, name)
    record["img_size"] = record["orig_size"]
    record["filename"] = os.path.join(root, record["filename"])
    record_pkl = pickle.dumps(record)

    return id, record_pkl


def process(annotation, root, name, split, target, max_size=2560, n_workers=8):
    annot = read_json(annotation)
    worker_fn = partial(
        worker, root=root, target=target, name=name, split=split, max_size=max_size
    )

    if name == "mlt19":
        refined_annot = []
        for r in annot[split]:
            sample = dict()
            sample["image_id"] = r["id"]
            sample["filename"] = r["file_name"].split("/")[-1]
            img = Image.open(os.path.join(annotation_path, root, sample["filename"]))
            sample["image_width"], sample["image_height"] = img.width, img.height
            words = []
            for quad, text in zip(r["QUAD"], r["TEXT"]):
                vertices = quad
                legible = text != "###"
                words.append({"vertices": vertices, "text": text, "legible": legible})
            sample["words"] = words
            refined_annot.append(sample)

        row = [(i, r) for i, r in enumerate(refined_annot)]

    elif name in ["icdar13", "icdar15"]:
        coco = COCO(annotation)
        refined_annot = []
        for img_annot in annot["images"]:
            img_id = img_annot["id"]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            sample = dict()
            sample["image_width"], sample["image_height"] = (
                img_annot["width"],
                img_annot["height"],
            )
            sample["image_id"] = img_annot["id"]
            sample["filename"] = img_annot["file_name"].split("/")[-1]
            words = []
            for ann in anns:
                assert len(ann["segmentation"]) == 1
                vertices = ann["segmentation"][0]
                text = ann["text"]
                legible = ann["text"] != "###"
                words.append({"vertices": vertices, "text": text, "legible": legible})
            sample["words"] = words
            refined_annot.append(sample)

        row = [(i, r) for i, r in enumerate(refined_annot)]

    elif name in [
        "synthtext150k.poly.part1",
        "synthtext150k.poly.part2",
        "totaltext.poly",
        "ctw1500.poly",
    ]:
        eos = len(synthtext_vocab)
        coco = COCO(annotation)
        refined_annot = []
        for img_annot in annot["images"]:
            img_id = img_annot["id"]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            sample = dict()
            sample["image_width"], sample["image_height"] = (
                img_annot["width"],
                img_annot["height"],
            )
            sample["image_id"] = img_annot["id"]
            sample["filename"] = img_annot["file_name"].split("/")[-1]
            words = []
            for ann in anns:
                vertices = ann["polys"]
                text = ""
                for idx in ann["rec"]:
                    if idx >= eos:
                        break
                    text += synthtext_vocab[idx]
                legible = text != "###" and text != ""
                words.append({"vertices": vertices, "text": text, "legible": legible})
            sample["words"] = words
            refined_annot.append(sample)

        row = [(i, r) for i, r in enumerate(refined_annot)]

    elif name == "hiertext":
        row = [(i, r) for i, r in enumerate(annot["annotations"])]

    elif name in ["textocr", "textocr.poly"]:
        refined_annot = []
        for img_id, ann_ids in annot["imgToAnns"].items():
            img_annot = annot["imgs"][img_id]
            sample = dict()
            sample["image_width"], sample["image_height"] = (
                img_annot["width"],
                img_annot["height"],
            )
            sample["image_id"] = img_id
            sample["filename"] = img_annot["file_name"].split("/")[-1]
            words = []
            for ann_id in ann_ids:
                vertices = annot["anns"][ann_id]["points"]
                text = annot["anns"][ann_id]["utf8_string"]
                legible = text != "."
                words.append({"vertices": vertices, "text": text, "legible": legible})
            sample["words"] = words
            refined_annot.append(sample)

        row = [(i, r) for i, r in enumerate(refined_annot)]

    with lmdb.open(
        os.path.join(target, name + "_" + split + ".lmdb"),
        map_size=1024 ** 4,
        readahead=False,
    ) as env, multiprocessing.Pool(n_workers) as pool:
        for i, record_pkl in tqdm(
            pool.imap_unordered(worker_fn, row),
            desc=os.path.basename(annotation),
            total=len(row),
            dynamic_ncols=True,
        ):
            with env.begin(write=True) as txn:
                txn.put(str(i).encode("utf-8"), record_pkl)

        with env.begin(write=True) as txn:
            txn.put(b"length", str(len(row)).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--max_size", type=int, default=2560)
    parser.add_argument("--out", type=str, default="./")
    parser.add_argument("--split", type=str, default="train")

    args = parser.parse_args()

    for i, annot in enumerate(annotation_files[args.split].keys(), start=1):
        print(f"{i}. {annot}")

    for dataset, (annot, root) in annotation_files[args.split].items():
        process(
            annot, root, dataset, args.split, args.out, args.max_size, args.n_workers
        )
