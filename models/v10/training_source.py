import os
import json
import csv
import time
import random
import unicodedata

from PIL import Image, ImageOps, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Sampler


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = (
    r"C:\ai-test\OCR-AJAMI-Hausa"
    r"\Hausa_repo_nouveau\data\train"
)

TEST_DIR = (
    r"C:\ai-test\OCR-AJAMI-Hausa"
    r"\Hausa_repo_nouveau\data\test"
)

SPLIT_PATH = (
    r"C:\ai-test\hausa_split_nouveau_92_3_5.json"
)

VOCAB_PATH = r"C:\ai-test\hausa_ctc_vocab_nouveau.json"

OUTPUT_DIR = (
    r"C:\ai-test\ctc_v10_nouveau_92_3_5"
)

BEST_CER_PATH = os.path.join(
    OUTPUT_DIR,
    "best_cer_v10.pt"
)

BEST_WER_PATH = os.path.join(
    OUTPUT_DIR,
    "best_wer_v10.pt"
)

LAST_PATH = os.path.join(
    OUTPUT_DIR,
    "last_v10.pt"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "history_v10.csv"
)


# ============================================================
# PARAMETRES PRINCIPAUX
# ============================================================

HEIGHT = 96

BATCH_SIZE = 24

BUCKET_SIZE = 192

# ------------------------------------------------------------
# Chaque image réelle produit :
#
# vue 0 = originale
# vues 1-7 = transformations fixes
#
# Donc :
#
# TRAIN virtuel = TRAIN réel × 8
# ------------------------------------------------------------

VIEWS_PER_IMAGE = 8


# ------------------------------------------------------------
# ATTENTION :
#
# Un epoch V10 correspond à 8 passages virtuels par image.
#
# On ne fait donc pas 100 epochs.
# ------------------------------------------------------------

MAX_EPOCHS = 25


INITIAL_LR = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42

VRAM_LIMIT_MB = 5500


# ============================================================
# EARLY STOP / SATURATION
# ============================================================

MIN_EPOCHS = 8

PATIENCE = 6

MIN_DELTA = 0.0005

TREND_WINDOW = 5

SLOPE_THRESHOLD = 0.0008

WINDOW_GAIN_THRESHOLD = 0.008


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCTIBILITE
# ============================================================

random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )

    torch.backends.cudnn.benchmark = True


# ============================================================
# VOCABULAIRE
# ============================================================

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:

    vocab = json.load(f)


BLANK_ID = vocab[
    "<BLANK>"
]


char_to_id = {
    c: i
    for c, i in vocab.items()
    if c != "<BLANK>"
}


id_to_char = {
    i: c
    for c, i in char_to_id.items()
}


NUM_CLASSES = len(
    vocab
)


# ============================================================
# CHARGEMENT DU SPLIT 92 / 3 / 5
# ============================================================

with open(
    SPLIT_PATH,
    "r",
    encoding="utf-8"
) as f:

    split = json.load(f)


train_entries = split[
    "train"
]

val_entries = split[
    "validation"
]

test_entries = split[
    "test"
]


# ============================================================
# VERIFICATION DU SPLIT
# ============================================================

TOTAL_SPLIT = (
    len(train_entries)
    + len(val_entries)
    + len(test_entries)
)


if TOTAL_SPLIT == 0:

    raise RuntimeError(
        "Le split est vide."
    )


train_ratio_real = (
    len(train_entries)
    / TOTAL_SPLIT
)

val_ratio_real = (
    len(val_entries)
    / TOTAL_SPLIT
)

test_ratio_real = (
    len(test_entries)
    / TOTAL_SPLIT
)


print("=" * 72)
print("SPLIT CHARGE")
print("=" * 72)

print(
    "TRAIN      :",
    len(train_entries),
    f"({train_ratio_real * 100:.2f} %)"
)

print(
    "VALIDATION :",
    len(val_entries),
    f"({val_ratio_real * 100:.2f} %)"
)

print(
    "TEST       :",
    len(test_entries),
    f"({test_ratio_real * 100:.2f} %)"
)

print(
    "TOTAL      :",
    TOTAL_SPLIT
)

print("=" * 72)


# ============================================================
# RESOLUTION DES CHEMINS
# ============================================================

def resolve_path(
    source,
    filename
):

    if source == "new_train":

        return os.path.join(
            TRAIN_DIR,
            filename
        )


    if source == "new_test":

        return os.path.join(
            TEST_DIR,
            filename
        )


    raise ValueError(
        "Source inconnue : "
        + str(source)
    )


# ============================================================
# AUGMENTATIONS FIXES
#
# 0 = originale
# 1 = luminosite / contraste
# 2 = epaisseur encre
# 3 = simulation scanner
# 4 = leger flou
# 5 = rotation
# 6 = deformation largeur/hauteur
# 7 = combinaison moderee
#
# IMPORTANT :
#
# Ces transformations sont DETERMINISTES.
#
# Une vue donnée reste identique à chaque epoch.
# ============================================================

def fixed_augmentation(
    image,
    variant,
    sample_index
):

    if variant == 0:

        return image


    rng = random.Random(
        SEED
        + sample_index * 10007
        + variant * 1000003
    )


    original_w, original_h = (
        image.size
    )


    # ========================================================
    # VUE 1
    #
    # Papier / luminosite / contraste
    # ========================================================

    if variant == 1:

        image = (
            ImageEnhance.Brightness(
                image
            ).enhance(
                rng.uniform(
                    0.88,
                    1.10
                )
            )
        )


        image = (
            ImageEnhance.Contrast(
                image
            ).enhance(
                rng.uniform(
                    0.82,
                    1.18
                )
            )
        )


        return image


    # ========================================================
    # VUE 2
    #
    # Epaisseur d'encre
    # ========================================================

    if variant == 2:

        if rng.random() < 0.5:

            image = image.filter(
                ImageFilter.MinFilter(
                    3
                )
            )

        else:

            image = image.filter(
                ImageFilter.MaxFilter(
                    3
                )
            )


        image = (
            ImageEnhance.Contrast(
                image
            ).enhance(
                rng.uniform(
                    0.92,
                    1.10
                )
            )
        )


        return image


    # ========================================================
    # VUE 3
    #
    # Simulation scanner
    # ========================================================

    if variant == 3:

        scale = rng.uniform(
            0.72,
            0.90
        )


        w2 = max(
            8,
            round(
                original_w
                * scale
            )
        )


        h2 = max(
            8,
            round(
                original_h
                * scale
            )
        )


        image = image.resize(
            (
                w2,
                h2
            ),
            Image.Resampling.BILINEAR
        )


        image = image.resize(
            (
                original_w,
                original_h
            ),
            Image.Resampling.BICUBIC
        )


        return image


    # ========================================================
    # VUE 4
    #
    # Leger flou optique
    # ========================================================

    if variant == 4:

        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=rng.uniform(
                    0.20,
                    0.55
                )
            )
        )


        image = (
            ImageEnhance.Contrast(
                image
            ).enhance(
                rng.uniform(
                    0.94,
                    1.08
                )
            )
        )


        return image


    # ========================================================
    # VUE 5
    #
    # Rotation legere
    # ========================================================

    if variant == 5:

        angle = rng.uniform(
            -1.0,
            1.0
        )


        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255
        )


        return image


    # ========================================================
    # VUE 6
    #
    # Variation geometrie
    # ========================================================

    if variant == 6:

        sx = rng.uniform(
            0.94,
            1.06
        )


        sy = rng.uniform(
            0.96,
            1.04
        )


        new_w = max(
            8,
            round(
                original_w
                * sx
            )
        )


        new_h = max(
            8,
            round(
                original_h
                * sy
            )
        )


        image = image.resize(
            (
                new_w,
                new_h
            ),
            Image.Resampling.BICUBIC
        )


        return image


    # ========================================================
    # VUE 7
    #
    # Document réel / combinaison moderee
    # ========================================================

    if variant == 7:

        image = (
            ImageEnhance.Brightness(
                image
            ).enhance(
                rng.uniform(
                    0.92,
                    1.07
                )
            )
        )


        image = (
            ImageEnhance.Contrast(
                image
            ).enhance(
                rng.uniform(
                    0.88,
                    1.14
                )
            )
        )


        if rng.random() < 0.50:

            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=rng.uniform(
                        0.12,
                        0.35
                    )
                )
            )


        angle = rng.uniform(
            -0.60,
            0.60
        )


        image = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255
        )


        return image


    return image


# ============================================================
# DATASET TRAIN VIRTUEL
# ============================================================

class FixedTrainDataset(
    Dataset
):

    def __init__(
        self,
        entries
    ):

        self.base_samples = []

        self.expected_widths = []


        for item in entries:

            source = item[
                "source"
            ]

            filename = item[
                "file_name"
            ]


            text = unicodedata.normalize(
                "NFC",
                item[
                    "transcript"
                ]
            )


            path = resolve_path(
                source,
                filename
            )


            if not os.path.exists(
                path
            ):

                raise FileNotFoundError(
                    path
                )


            unknown = [
                c
                for c in text
                if c not in char_to_id
            ]


            if unknown:

                raise RuntimeError(
                    f"Caracteres inconnus "
                    f"dans {filename}: "
                    f"{unknown}"
                )


            with Image.open(
                path
            ) as img:

                w, h = img.size


            expected_width = max(
                4,
                round(
                    w
                    * HEIGHT
                    / h
                )
            )


            self.base_samples.append(
                (
                    path,
                    text,
                    filename,
                    source,
                    expected_width
                )
            )


        # ====================================================
        # Largeurs approx pour bucketing
        # ====================================================

        for base_index in range(
            len(self.base_samples)
        ):

            expected = (
                self.base_samples[
                    base_index
                ][4]
            )


            for _ in range(
                VIEWS_PER_IMAGE
            ):

                self.expected_widths.append(
                    expected
                )


    def __len__(
        self
    ):

        return (
            len(
                self.base_samples
            )
            * VIEWS_PER_IMAGE
        )


    def __getitem__(
        self,
        index
    ):

        base_index = (
            index
            // VIEWS_PER_IMAGE
        )


        variant = (
            index
            % VIEWS_PER_IMAGE
        )


        (
            path,
            text,
            filename,
            source,
            _
        ) = self.base_samples[
            base_index
        ]


        image = Image.open(
            path
        ).convert(
            "L"
        )


        image = fixed_augmentation(
            image,
            variant,
            base_index
        )


        # ====================================================
        # REDIMENSIONNEMENT
        # ====================================================

        w, h = image.size


        new_w = max(
            4,
            round(
                w
                * HEIGHT
                / h
            )
        )


        image = image.resize(
            (
                new_w,
                HEIGHT
            ),
            Image.Resampling.LANCZOS
        )


        # ====================================================
        # RTL AJAMI -> LTR CTC
        # ====================================================

        image = ImageOps.mirror(
            image
        )


        # ====================================================
        # TENSOR
        # ====================================================

        data = torch.frombuffer(
            bytearray(
                image.tobytes()
            ),
            dtype=torch.uint8
        )


        data = data.reshape(
            HEIGHT,
            new_w
        ).float()


        data /= 255.0


        data = (
            data
            - 0.5
        ) / 0.5


        data = data.unsqueeze(
            0
        )


        target = torch.tensor(
            [
                char_to_id[c]
                for c in text
            ],
            dtype=torch.long
        )


        return (
            data,
            target,
            text,
            new_w,
            filename,
            source,
            variant
        )


# ============================================================
# DATASET VAL / TEST
#
# AUCUNE AUGMENTATION
# ============================================================

class CleanDataset(
    Dataset
):

    def __init__(
        self,
        entries
    ):

        self.samples = []


        for item in entries:

            source = item[
                "source"
            ]

            filename = item[
                "file_name"
            ]


            text = unicodedata.normalize(
                "NFC",
                item[
                    "transcript"
                ]
            )


            path = resolve_path(
                source,
                filename
            )


            if not os.path.exists(
                path
            ):

                raise FileNotFoundError(
                    path
                )


            unknown = [
                c
                for c in text
                if c not in char_to_id
            ]


            if unknown:

                raise RuntimeError(
                    f"Caracteres inconnus "
                    f"dans {filename}: "
                    f"{unknown}"
                )


            self.samples.append(
                (
                    path,
                    text,
                    filename,
                    source
                )
            )


    def __len__(
        self
    ):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        (
            path,
            text,
            filename,
            source
        ) = self.samples[
            index
        ]


        image = Image.open(
            path
        ).convert(
            "L"
        )


        w, h = image.size


        new_w = max(
            4,
            round(
                w
                * HEIGHT
                / h
            )
        )


        image = image.resize(
            (
                new_w,
                HEIGHT
            ),
            Image.Resampling.LANCZOS
        )


        image = ImageOps.mirror(
            image
        )


        data = torch.frombuffer(
            bytearray(
                image.tobytes()
            ),
            dtype=torch.uint8
        )


        data = data.reshape(
            HEIGHT,
            new_w
        ).float()


        data /= 255.0


        data = (
            data
            - 0.5
        ) / 0.5


        data = data.unsqueeze(
            0
        )


        target = torch.tensor(
            [
                char_to_id[c]
                for c in text
            ],
            dtype=torch.long
        )


        return (
            data,
            target,
            text,
            new_w,
            filename,
            source,
            0
        )


# ============================================================
# COLLATE
# ============================================================

def collate_fn(
    batch
):

    images = []

    targets = []

    texts = []

    widths = []

    filenames = []

    sources = []

    variants = []


    for (
        image,
        target,
        text,
        width,
        filename,
        source,
        variant
    ) in batch:

        images.append(
            image
        )

        targets.append(
            target
        )

        texts.append(
            text
        )

        widths.append(
            width
        )

        filenames.append(
            filename
        )

        sources.append(
            source
        )

        variants.append(
            variant
        )


    max_width = max(
        image.shape[-1]
        for image in images
    )


    padded = []


    for image in images:

        pad_width = (
            max_width
            - image.shape[-1]
        )


        image = F.pad(
            image,
            (
                0,
                pad_width,
                0,
                0
            ),
            value=1.0
        )


        padded.append(
            image
        )


    return (

        torch.stack(
            padded
        ),

        torch.cat(
            targets
        ),

        torch.tensor(
            [
                len(t)
                for t in targets
            ],
            dtype=torch.long
        ),

        texts,

        torch.tensor(
            widths,
            dtype=torch.long
        ),

        filenames,

        sources,

        variants
    )


# ============================================================
# BUCKET SAMPLER
# ============================================================

class WidthBucketSampler(
    Sampler
):

    def __init__(
        self,
        dataset,
        batch_size,
        bucket_size,
        seed
    ):

        self.dataset = dataset

        self.batch_size = batch_size

        self.bucket_size = (
            bucket_size
        )

        self.seed = seed

        self.epoch = 0


    def set_epoch(
        self,
        epoch
    ):

        self.epoch = epoch


    def __len__(
        self
    ):

        return (
            len(self.dataset)
            + self.batch_size
            - 1
        ) // self.batch_size


    def __iter__(
        self
    ):

        rng = random.Random(
            self.seed
            + self.epoch
        )


        indices = list(
            range(
                len(self.dataset)
            )
        )


        rng.shuffle(
            indices
        )


        batches = []


        for start in range(
            0,
            len(indices),
            self.bucket_size
        ):

            bucket = indices[
                start:
                start
                + self.bucket_size
            ]


            bucket.sort(
                key=lambda idx:
                    self.dataset.expected_widths[
                        idx
                    ]
            )


            for pos in range(
                0,
                len(bucket),
                self.batch_size
            ):

                batch = bucket[
                    pos:
                    pos
                    + self.batch_size
                ]


                if batch:

                    batches.append(
                        batch
                    )


        rng.shuffle(
            batches
        )


        for batch in batches:

            yield batch


# ============================================================
# CRNN
#
# IDENTIQUE A V8 / V9 / V10 ORIGINAL
# ============================================================

class CRNN(
    nn.Module
):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()


        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                (
                    2,
                    2
                )
            ),


            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                (
                    2,
                    2
                )
            ),


            nn.Conv2d(
                128,
                256,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                (
                    2,
                    1
                )
            ),


            nn.Conv2d(
                256,
                256,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                (
                    2,
                    1
                )
            ),


            nn.Conv2d(
                256,
                384,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                384
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                (
                    2,
                    1
                )
            ),


            nn.Conv2d(
                384,
                384,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                384
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                (
                    1,
                    None
                )
            )
        )


        self.rnn = nn.LSTM(
            input_size=384,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )


        self.classifier = nn.Linear(
            512,
            num_classes
        )


    def forward(
        self,
        x
    ):

        x = self.cnn(
            x
        )


        x = x.squeeze(
            2
        )


        x = x.permute(
            0,
            2,
            1
        )


        x, _ = self.rnn(
            x
        )


        return self.classifier(
            x
        )


# ============================================================
# CTC LENGTH
# ============================================================

def output_lengths(
    widths
):

    return (
        widths
        // 2
        // 2
    )


# ============================================================
# DECODAGE CTC GREEDY
# ============================================================

def decode_ctc(
    ids
):

    chars = []

    previous = None


    for idx in ids:

        idx = int(
            idx
        )


        if (
            idx != BLANK_ID
            and idx != previous
        ):

            chars.append(
                id_to_char.get(
                    idx,
                    ""
                )
            )


        previous = idx


    return "".join(
        chars
    )


# ============================================================
# LEVENSHTEIN
# ============================================================

def edit_distance(
    a,
    b
):

    previous = list(
        range(
            len(b)
            + 1
        )
    )


    for i, aa in enumerate(
        a,
        1
    ):

        current = [
            i
        ]


        for j, bb in enumerate(
            b,
            1
        ):

            current.append(
                min(
                    current[
                        j - 1
                    ] + 1,

                    previous[
                        j
                    ] + 1,

                    previous[
                        j - 1
                    ]
                    + (
                        aa != bb
                    )
                )
            )


        previous = current


    return previous[
        -1
    ]


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    max_examples=5
):

    model.eval()


    char_errors = 0

    char_total = 0


    word_errors = 0

    word_total = 0


    exact = 0

    total_lines = 0


    examples = []


    for batch in loader:

        (
            images,
            targets,
            target_lengths,
            texts,
            widths,
            filenames,
            sources,
            variants
        ) = batch


        images = images.to(
            DEVICE,
            non_blocking=True
        )


        logits = model(
            images
        )


        predictions = logits.argmax(
            dim=-1
        ).cpu()


        lengths = output_lengths(
            widths
        )


        for i, reference in enumerate(
            texts
        ):

            prediction = decode_ctc(
                predictions[
                    i,
                    :int(
                        lengths[i]
                    )
                ]
            )


            reference = (
                unicodedata.normalize(
                    "NFC",
                    reference
                )
            )


            prediction = (
                unicodedata.normalize(
                    "NFC",
                    prediction
                )
            )


            char_errors += (
                edit_distance(
                    reference,
                    prediction
                )
            )


            char_total += len(
                reference
            )


            ref_words = (
                reference.split()
            )


            pred_words = (
                prediction.split()
            )


            word_errors += (
                edit_distance(
                    ref_words,
                    pred_words
                )
            )


            word_total += len(
                ref_words
            )


            if prediction == reference:

                exact += 1


            total_lines += 1


            if (
                len(examples)
                < max_examples
            ):

                examples.append(
                    (
                        filenames[i],
                        reference,
                        prediction
                    )
                )


    cer = (
        char_errors
        / char_total
        if char_total > 0
        else 0.0
    )


    wer = (
        word_errors
        / word_total
        if word_total > 0
        else 0.0
    )


    exact_rate = (
        exact
        / total_lines
        if total_lines > 0
        else 0.0
    )


    return (
        cer,
        wer,
        exact_rate,
        examples
    )


# ============================================================
# PENTE LINEAIRE
# ============================================================

def linear_slope(
    values
):

    n = len(
        values
    )


    if n < 2:

        return 0.0


    x_mean = (
        n - 1
    ) / 2.0


    y_mean = (
        sum(values)
        / n
    )


    numerator = 0.0

    denominator = 0.0


    for i, value in enumerate(
        values
    ):

        dx = (
            i
            - x_mean
        )


        numerator += (
            dx
            * (
                value
                - y_mean
            )
        )


        denominator += (
            dx
            * dx
        )


    if denominator == 0:

        return 0.0


    return (
        numerator
        / denominator
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    val_cer,
    val_wer,
    val_exact
):

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_cer":
                val_cer,

            "val_wer":
                val_wer,

            "val_exact_rate":
                val_exact,

            "version":
                "V10_NOUVEAU_92_3_5",

            "height":
                HEIGHT,

            "batch_size":
                BATCH_SIZE,

            "base_train_images":
                len(
                    train_entries
                ),

            "views_per_image":
                VIEWS_PER_IMAGE,

            "virtual_train_size":
                len(
                    train_entries
                )
                * VIEWS_PER_IMAGE,

            "dynamic_augmentation":
                False,

            "fixed_augmentation":
                True,

            "rtl_mirror":
                True,

            "from_scratch":
                True,

            "split":
                "92_3_5",

            "split_path":
                SPLIT_PATH,

            "vocab":
                vocab
        },
        path
    )


# ============================================================
# DATASETS
# ============================================================

print()
print("=" * 72)
print("CREATION DATASET VIRTUEL V10")
print("=" * 72)


train_dataset = (
    FixedTrainDataset(
        train_entries
    )
)


val_dataset = (
    CleanDataset(
        val_entries
    )
)


test_dataset = (
    CleanDataset(
        test_entries
    )
)


# ============================================================
# SAMPLER
# ============================================================

train_sampler = (
    WidthBucketSampler(
        train_dataset,
        BATCH_SIZE,
        BUCKET_SIZE,
        SEED
    )
)


# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=True
)


# ============================================================
# MODELE
# ============================================================

model = CRNN(
    NUM_CLASSES
).to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CTCLoss(
    blank=BLANK_ID,
    reduction="mean",
    zero_infinity=True
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=INITIAL_LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
)


# ============================================================
# INFO
# ============================================================

print()
print("=" * 72)
print("HAUSA CTC V10 - NOUVEAU DATASET 92/3/5")
print("=" * 72)


print(
    "Device               :",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU                  :",
        torch.cuda.get_device_name(
            0
        )
    )


print(
    "Train reel           :",
    len(train_entries)
)

print(
    "Validation           :",
    len(val_entries)
)

print(
    "Test                 :",
    len(test_entries)
)

print(
    "Vues / image train   :",
    VIEWS_PER_IMAGE
)

print(
    "Train virtuel        :",
    len(train_dataset)
)

print(
    "Height               :",
    HEIGHT
)

print(
    "Batch                :",
    BATCH_SIZE
)

print(
    "Augmentation fixe    : ON"
)

print(
    "Augmentation dynamique:",
    "OFF"
)

print(
    "Initialisation       : FROM SCRATCH"
)

print(
    "Epochs max           :",
    MAX_EPOCHS
)

print(
    "LR initial           :",
    INITIAL_LR
)

print(
    "VRAM limite          :",
    VRAM_LIMIT_MB,
    "MB"
)

print("=" * 72)


# ============================================================
# HISTORIQUE CSV
# ============================================================

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )


    writer.writerow(
        [
            "epoch",
            "train_loss",
            "val_cer",
            "val_wer",
            "val_exact",
            "slope",
            "window_gain",
            "lr",
            "seconds",
            "vram_mb"
        ]
    )


# ============================================================
# ETAT
# ============================================================

best_cer = float(
    "inf"
)

best_wer = float(
    "inf"
)


best_cer_epoch = 0

best_wer_epoch = 0


best_for_patience = float(
    "inf"
)


without_improvement = 0


cer_history = []


start_total = time.time()


# ============================================================
# TRAIN
# ============================================================

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    train_sampler.set_epoch(
        epoch
    )


    model.train()


    running_loss = 0.0

    batches = 0


    start_epoch = time.time()


    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()


    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch in train_loader:

        (
            images,
            targets,
            target_lengths,
            texts,
            widths,
            filenames,
            sources,
            variants
        ) = batch


        images = images.to(
            DEVICE,
            non_blocking=True
        )


        targets = targets.to(
            DEVICE,
            non_blocking=True
        )


        target_lengths = (
            target_lengths.to(
                DEVICE,
                non_blocking=True
            )
        )


        input_lengths = (
            output_lengths(
                widths
            ).to(
                DEVICE
            )
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            images
        )


        log_probs = F.log_softmax(
            logits,
            dim=-1
        ).permute(
            1,
            0,
            2
        )


        loss = criterion(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
        )


        batches += 1


    # ========================================================
    # TRAIN LOSS
    # ========================================================

    train_loss = (
        running_loss
        / max(
            1,
            batches
        )
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_cer,
        val_wer,
        val_exact,
        examples
    ) = evaluate(
        model,
        val_loader,
        5
    )


    cer_history.append(
        val_cer
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_cer
    )


    lr = (
        optimizer.param_groups[
            0
        ]["lr"]
    )


    # ========================================================
    # TENDANCE
    # ========================================================

    if (
        len(cer_history)
        >= TREND_WINDOW
    ):

        recent = cer_history[
            -TREND_WINDOW:
        ]


        slope = linear_slope(
            recent
        )


        window_gain = (
            recent[0]
            - min(
                recent
            )
        )

    else:

        slope = 0.0

        window_gain = float(
            "inf"
        )


    # ========================================================
    # BEST CER
    # ========================================================

    if val_cer < best_cer:

        best_cer = val_cer

        best_cer_epoch = (
            epoch
        )


        save_checkpoint(
            BEST_CER_PATH,
            epoch,
            model,
            optimizer,
            val_cer,
            val_wer,
            val_exact
        )


        print()
        print(
            ">>> NOUVEAU BEST CER"
        )


    # ========================================================
    # BEST WER
    # ========================================================

    if val_wer < best_wer:

        best_wer = val_wer

        best_wer_epoch = (
            epoch
        )


        save_checkpoint(
            BEST_WER_PATH,
            epoch,
            model,
            optimizer,
            val_cer,
            val_wer,
            val_exact
        )


        print()
        print(
            ">>> NOUVEAU BEST WER"
        )


    # ========================================================
    # PATIENCE
    # ========================================================

    if (
        val_cer
        < best_for_patience
        - MIN_DELTA
    ):

        best_for_patience = (
            val_cer
        )


        without_improvement = 0

    else:

        without_improvement += 1


    # ========================================================
    # TEMPS
    # ========================================================

    seconds = (
        time.time()
        - start_epoch
    )


    # ========================================================
    # VRAM
    # ========================================================

    if torch.cuda.is_available():

        vram_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    else:

        vram_mb = 0.0


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 72)

    print(
        f"EPOCH {epoch:03d}/{MAX_EPOCHS}"
    )

    print("=" * 72)


    print(
        f"Loss train  : {train_loss:.4f}"
    )

    print(
        f"VAL CER     : {val_cer:.4f}"
    )

    print(
        f"VAL WER     : {val_wer:.4f}"
    )

    print(
        f"VAL exact   : {val_exact:.4f}"
    )


    if (
        len(cer_history)
        >= TREND_WINDOW
    ):

        print(
            f"Pente CER   : "
            f"{slope:+.6f}/epoch"
        )

        print(
            f"Gain fenetre: "
            f"{window_gain:.4f}"
        )

    else:

        print(
            "Pente CER   : attente"
        )


    print(
        f"LR          : {lr:.8f}"
    )

    print(
        f"Temps       : {seconds:.1f} sec"
    )

    print(
        f"VRAM alloc  : {vram_mb:.0f} MB"
    )


    print(
        "Train reel  :",
        len(train_entries)
    )

    print(
        "Vues epoch  :",
        len(train_dataset)
    )

    print(
        "Equivalent  :",
        f"x{VIEWS_PER_IMAGE}"
    )


    if (
        vram_mb
        > VRAM_LIMIT_MB
    ):

        print()

        print(
            "!!! ALERTE VRAM !!!"
        )


    # ========================================================
    # EXEMPLES
    # ========================================================

    for (
        filename,
        reference,
        prediction
    ) in examples:

        print()

        print(
            "FILE:",
            filename
        )

        print(
            "REF :",
            reference
        )

        print(
            "PRED:",
            prediction
        )


    # ========================================================
    # BEST INFO
    # ========================================================

    print()


    print(
        "Best CER    :",
        round(
            best_cer,
            4
        ),
        "@",
        best_cer_epoch
    )


    print(
        "Best WER    :",
        round(
            best_wer,
            4
        ),
        "@",
        best_wer_epoch
    )


    print(
        "Sans gain   :",
        without_improvement,
        "/",
        PATIENCE
    )


    # ========================================================
    # CSV
    # ========================================================

    with open(
        HISTORY_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                epoch,
                train_loss,
                val_cer,
                val_wer,
                val_exact,
                slope,
                window_gain,
                lr,
                seconds,
                vram_mb
            ]
        )


    # ========================================================
    # LAST
    # ========================================================

    save_checkpoint(
        LAST_PATH,
        epoch,
        model,
        optimizer,
        val_cer,
        val_wer,
        val_exact
    )


    # ========================================================
    # SATURATION
    # ========================================================

    saturation = (

        epoch
        >= MIN_EPOCHS

        and

        len(
            cer_history
        )
        >= TREND_WINDOW

        and

        abs(
            slope
        )
        < SLOPE_THRESHOLD

        and

        window_gain
        < WINDOW_GAIN_THRESHOLD

        and

        without_improvement
        >= PATIENCE
    )


    if saturation:

        print()

        print("=" * 72)

        print(
            "SATURATION DETECTEE"
        )

        print(
            "Best epoch :",
            best_cer_epoch
        )

        print(
            "Best CER   :",
            round(
                best_cer,
                4
            )
        )

        print(
            "Pente      :",
            round(
                slope,
                6
            )
        )

        print(
            "Gain       :",
            round(
                window_gain,
                4
            )
        )

        print("=" * 72)

        break


# ============================================================
# TEST FINAL
#
# On recharge le BEST CER.
# ============================================================

print()

print("=" * 72)

print(
    "TEST FINAL"
)

print("=" * 72)


checkpoint = torch.load(
    BEST_CER_PATH,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


(
    test_cer,
    test_wer,
    test_exact,
    examples
) = evaluate(
    model,
    test_loader,
    10
)


minutes = (
    time.time()
    - start_total
) / 60


# ============================================================
# RESUME FINAL
# ============================================================

print()
print("=" * 72)

print(
    "RESUME FINAL V10"
)

print("=" * 72)


print(
    "BEST epoch       :",
    checkpoint[
        "epoch"
    ]
)

print(
    "BEST VAL CER     :",
    round(
        checkpoint[
            "val_cer"
        ],
        4
    )
)

print(
    "BEST VAL WER     :",
    round(
        checkpoint[
            "val_wer"
        ],
        4
    )
)

print(
    "BEST VAL exact   :",
    round(
        checkpoint[
            "val_exact_rate"
        ],
        4
    )
)


print()

print(
    "TEST CER         :",
    round(
        test_cer,
        4
    )
)

print(
    "TEST WER         :",
    round(
        test_wer,
        4
    )
)

print(
    "TEST exact rate  :",
    round(
        test_exact,
        4
    )
)


print()

print(
    "Train réel       :",
    len(
        train_entries
    )
)

print(
    "Train virtuel    :",
    len(
        train_dataset
    )
)

print(
    "Vues / image     :",
    VIEWS_PER_IMAGE
)


print()

print(
    "Temps total      :",
    round(
        minutes,
        1
    ),
    "minutes"
)


print()

print(
    "Best CER :",
    BEST_CER_PATH
)

print(
    "Best WER :",
    BEST_WER_PATH
)

print(
    "History  :",
    HISTORY_PATH
)


# ============================================================
# EXEMPLES TEST
# ============================================================

print()

print("=" * 72)

print(
    "EXEMPLES TEST"
)

print("=" * 72)


for (
    filename,
    reference,
    prediction
) in examples:

    print()

    print(
        "FILE:",
        filename
    )

    print(
        "REF :",
        reference
    )

    print(
        "PRED:",
        prediction
    )