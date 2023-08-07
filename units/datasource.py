import os

from PIL import Image
from tensorfn.data import LMDBReader

from units.structures import OCRInstances, Sample


class LMDBSource:
    task_key = "ocr"

    def __init__(self, root, annotation):
        """
        Args:
            root (str): Root path indicates the directory contains image and lmdb
            annotation (str): Path to the annotation lmdb relative to root
        """
        self.root = root
        self.annots = LMDBReader(os.path.join(root, annotation), reader="pickle")
        self.key = os.path.splitext(annotation)[0]

    def __len__(self):
        return len(self.annots)

    def read_image(self, path):
        img = Image.open(os.path.join(self.root, path))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __getitem__(self, index):
        """
        Returns:
            img (Image): Raw pillow image of the record
            sample (Sample): Sample with ocr fields, which contains:
                coords (List[List[Tuple[float, float]]]):
                    (x, y) coordinate of bounding polygon of each entries
                texts (List[str]): Text content of each entries

        !Important! text with length 0 ('') indicates don't care area!
        """

        annot = self.annots[index]

        words = annot["words"]
        dcs = annot["dcs"]
        img_path = annot["filename"]
        orig_size = annot["orig_size"]

        img = self.read_image(img_path)

        coords = []
        texts = []

        for word in words:
            points = word[0]
            letters = word[1]

            coords.append(points)
            texts.append(letters)

        for dc in dcs:
            coords.append(dc)
            texts.append("")

        return img, Sample(
            image_size=img.size[::-1],
            orig_size=orig_size[::-1],
            img_path=img_path,
            key=self.key,
            ocr=OCRInstances(
                coords,
                texts,
            ),
        )
