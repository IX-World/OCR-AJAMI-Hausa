"""IX-World: memory-bounded CRNN with an auxiliary contextual text teacher."""
import math
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .models import CRNN6


class SemanticCRNN(nn.Module):
    def __init__(self, num_classes, context_schema, semantic_dim=768):
        super().__init__()
        base = CRNN6(num_classes, input_channels=1)
        layers = []
        for layer in base.cnn:
            if isinstance(layer, nn.BatchNorm2d):
                layer = nn.GroupNorm(8, layer.num_features)
            layers.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                layers.append(nn.Dropout2d(0.1))
        self.cnn = nn.Sequential(*layers)
        self.rnn = base.rnn
        self.classifier = base.classifier
        self.language = nn.Embedding(len(context_schema["languages"]), 16, padding_idx=0)
        self.locality = nn.Embedding(len(context_schema["localities"]), 16, padding_idx=0)
        self.time = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        self.film = nn.Linear(48, 1024, bias=False)
        self.sequence_dropout = nn.Dropout(0.2)
        self.attention = nn.Linear(512, 1)
        self.semantic = nn.Sequential(nn.Linear(512, semantic_dim), nn.LayerNorm(semantic_dim))

    def forward(self, images, widths, context, return_semantic=False):
        lengths = widths.detach().to("cpu", dtype=torch.long) // 4
        features = self.cnn(images).squeeze(2).transpose(1, 2)
        if int(lengths.min()) < 1 or int(lengths.max()) > features.shape[1]:
            raise ValueError("Longueurs CTC invalides.")
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        sequence, _ = self.rnn(packed)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=features.shape[1])
        temporal = self.time(context["time"][:, 1:]) * context["time"][:, :1]
        metadata = torch.cat((self.language(context["language"]), self.locality(context["locality"]), temporal), -1)
        gamma, beta = (0.1 * self.film(metadata).tanh()).chunk(2, dim=-1)
        sequence = sequence * (1 + gamma[:, None]) + beta[:, None]
        logits = self.classifier(self.sequence_dropout(sequence))
        if not return_semantic:
            return logits
        valid = torch.arange(sequence.shape[1], device=sequence.device)[None, :] < lengths.to(sequence.device)[:, None]
        weights = self.attention(sequence).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        pooled = (sequence * weights[..., None]).sum(1)
        return logits, self.semantic(pooled)


class PixelBudgetSampler:
    """Microbatches limited by count AND padded pixels; no image truncation."""
    def __init__(self, dataset, batch_size=6, max_pixels=600000, seed=42, training=False):
        if batch_size < 1 or max_pixels < 96:
            raise ValueError("Budget de batch invalide.")
        self.dataset, self.batch_size, self.max_pixels = dataset, batch_size, max_pixels
        self.seed, self.training, self.epoch = seed, training, 0
        # Fixed rotations/scales can enlarge the width slightly.
        self.widths = [math.ceil(w * 1.2) + 32 if training else w for w in dataset.expected_widths]
        if max(self.widths) * 96 > max_pixels:
            raise ValueError("Une image depasse le budget de pixels ; augmenter max-batch-pixels sans recadrer le texte.")

    def batches(self):
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self.epoch)
        if self.training:
            rng.shuffle(indices)
            bucket_size = self.batch_size * 32
            indices = [i for start in range(0, len(indices), bucket_size)
                       for i in sorted(indices[start:start + bucket_size], key=lambda j: self.widths[j])]
        result, current, width = [], [], 0
        for index in indices:
            proposed = max(width, self.widths[index])
            if current and (len(current) >= self.batch_size or (len(current) + 1) * proposed * 96 > self.max_pixels):
                result.append(current)
                current, width = [], 0
            current.append(index)
            width = max(width, self.widths[index])
        if current:
            result.append(current)
        if self.training:
            rng.shuffle(result)
        return result

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())
