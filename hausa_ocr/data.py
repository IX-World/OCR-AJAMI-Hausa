import math
import random

import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset, Sampler

from . import augmentations


class LineDataset(Dataset):
    def __init__(self, root, entries, vocab, version, training=False, seed=42):
        self.root, self.entries, self.vocab = root, entries, vocab
        self.version, self.training, self.seed = version, training, seed
        self.views = 8 if training and version != "v9" else 1
        self.epoch = 0
        self.expected_widths = [max(4, round(row["width"] * 96 / row["height"])) for row in entries for _ in range(self.views)]

    def __len__(self):
        return len(self.entries) * self.views

    def __getitem__(self, index):
        base, view = divmod(index, self.views)
        row = self.entries[base]
        with Image.open(self.root / "data" / row["partition"] / row["file_name"]) as original:
            image = original.convert("L")
        if self.training and self.version == "v9":
            rng = random.Random(self.seed + self.epoch * 1000003 + base * 9176)
            if rng.random() < 0.30:
                image = augmentations.augment_image(image, rng)
        elif self.training:
            augmentations.SEED = self.seed
            image = augmentations.fixed_augmentation(image, view, base)
        width = max(4, round(image.width * 96 / image.height))
        image = ImageOps.mirror(image.resize((width, 96), Image.Resampling.LANCZOS))
        gray = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(96, width).float() / 255.0
        channels = [gray * 2 - 1]
        if self.version in ("v11", "v12"):
            background = image.filter(ImageFilter.GaussianBlur(radius=5.0))
            background = torch.frombuffer(bytearray(background.tobytes()), dtype=torch.uint8).reshape(96, width).float() / 255.0
            ink = ((background - gray).clamp(min=0) * 1.6).clamp(0, 1)
            channels.append(ink * 2 - 1)
        target = torch.tensor([self.vocab.get(c, self.vocab["<UNK>"]) for c in row["transcript"]], dtype=torch.long)
        if self.training:
            minimum = len(target) + sum(a == b for a, b in zip(row["transcript"], row["transcript"][1:]))
            if width // 4 < minimum:
                raise ValueError(f"Alignement CTC impossible apres augmentation : {row['file_name']}, vue {view}")
        return torch.stack(channels), target, row, width


def collate(batch):
    channels = batch[0][0].shape[0]
    width = max(item[3] for item in batch)
    images = torch.full((len(batch), channels, 96, width), -1.0)
    images[:, 0].fill_(1.0)
    for i, (image, _, _, item_width) in enumerate(batch):
        images[i, :, :, :item_width] = image
    return {
        "images": images,
        "targets": torch.cat([item[1] for item in batch]),
        "target_lengths": torch.tensor([len(item[1]) for item in batch], dtype=torch.long),
        "widths": torch.tensor([item[3] for item in batch], dtype=torch.long),
        "rows": [item[2] for item in batch],
    }


class WidthSampler(Sampler):
    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
        self.bucket_size = batch_size * 8

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = sorted(indices[start:start + self.bucket_size], key=lambda i: self.dataset.expected_widths[i])
            batches.extend(bucket[pos:pos + self.batch_size] for pos in range(0, len(bucket), self.batch_size))
        rng.shuffle(batches)
        yield from batches


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def decode(ids, vocab):
    inverse = {value: key for key, value in vocab.items()}
    chars, previous = [], None
    for value in ids:
        value = int(value)
        if value and value != previous:
            chars.append("\ufffd" if value == vocab["<UNK>"] else inverse[value])
        previous = value
    return "".join(chars)
