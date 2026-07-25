import os
import re
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # lets PIL.Image.open() read .heic directly
except ImportError:
    pass


def _numeric_sort_key(filename):
    nums = re.findall(r"\d+", filename)
    return (int(nums[0]) if nums else float("inf"), filename)


def _discover_folder_based(images_dir):
    valid_ext = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp")
    classes = sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )
    image_paths = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(images_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_ext)]
        files.sort(key=_numeric_sort_key)
        for f in files:
            image_paths.append(os.path.join(cls_dir, f))
            labels.append(cls)
    return image_paths, labels, classes


class MedicineDataset(Dataset):
    def __init__(self, images_dir, labels_file=None, transform=None, label_list=None, folder_based=False):
        self.images_dir = images_dir
        self.transform = transform

        if folder_based or labels_file is None:
            self.image_paths, self.labels_raw, self.classes = _discover_folder_based(images_dir)
        else:
            valid_ext = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp")
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
            image_files.sort(key=_numeric_sort_key)

            with open(labels_file, "r", encoding="utf-8") as f:
                labels_raw = [line.strip() for line in f if line.strip()]

            if len(image_files) != len(labels_raw):
                raise ValueError(
                    f"Mismatch: found {len(image_files)} images in '{images_dir}' "
                    f"but {len(labels_raw)} label lines in '{labels_file}'. "
                    f"Each image must have exactly one matching line, in order.\n"
                    f"Images (sorted): {image_files}\n"
                    f"Labels: {labels_raw}"
                )

            self.image_paths = [os.path.join(images_dir, f) for f in image_files]
            self.labels_raw = labels_raw
            self.classes = label_list if label_list is not None else sorted(set(labels_raw))

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels_raw]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def save_classes(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.classes, f, indent=2)
