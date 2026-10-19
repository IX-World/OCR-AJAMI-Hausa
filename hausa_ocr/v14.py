"""IX-World: V12-compatible augmentation, bounded batches, SAM and MWER."""
import contextlib
import itertools
import math
import random

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

from . import augmentations
from .ctc_beam import prefix_beam_search, tokens_to_text
from .data import edit_distance


V12_CHECKPOINT_SHA256 = "99a0d0f2c824a242b1b5e4864a5af2f4df0c291064ad0f1eba2ddeb6d5c3f6c1"
SPLIT_SHA256 = "fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5"
VOCAB_SHA256 = "f92c368520aad95130f909a708fb58864e1f766f10f87d18c96690d1b20ced59"
COUNTS = {"train": 4972, "validation": 200, "test": 401}
TRIALS = {
    "A": {"augmentation": "fixed_8_views", "sam": False, "mwer": False},
    "B": {"augmentation": "renewed_8_views", "sam": False, "mwer": False},
    "C": {"augmentation": "renewed_8_views", "sam": True, "mwer": False},
    "D": {"augmentation": "renewed_8_views", "sam": True, "mwer": True},
}


def require_protocol(report):
    if (report["split_sha256"] != SPLIT_SHA256 or report["vocab_sha256"] != VOCAB_SHA256 or report["counts"] != COUNTS or
            not report.get("all_usable_images_used") or report.get("separation_unit") != "source_and_known_scribe"):
        raise ValueError("V14 exige le split commun fige 4972/200/401 et son vocabulaire.")


class TrialDataset(Dataset):
    def __init__(self, root, entries, vocab, policy="originals", seed=42):
        if policy not in ("originals", "fixed_8_views", "renewed_8_views"):
            raise ValueError("Politique d'augmentation inconnue.")
        if policy != "originals" and any(r.get("split") != "train" for r in entries):
            raise ValueError("Les augmentations V14 sont reservees au TRAIN experimental.")
        self.root, self.entries, self.vocab, self.policy, self.seed = root, entries, vocab, policy, seed
        self.views = 1 if policy == "originals" else 8
        self.epoch = 0
        self.expected_widths = [max(4, round(r["width"] * 96 / r["height"])) for r in entries for _ in range(self.views)]

    def __len__(self):
        return len(self.entries) * self.views

    def __getitem__(self, index):
        base, view = divmod(index, self.views)
        row = self.entries[base]
        with Image.open(self.root / "data" / row["partition"] / row["file_name"]) as original:
            image = original.convert("L")
        if self.policy != "originals":
            # All loaders use workers=0. Restore the historical module's seed.
            previous_seed = augmentations.SEED
            try:
                augmentations.SEED = self.seed + (self.epoch * 1000000007 if self.policy == "renewed_8_views" else 0)
                image = augmentations.fixed_augmentation(image, view, base)
            finally:
                augmentations.SEED = previous_seed
        width = max(4, round(image.width * 96 / image.height))
        image = ImageOps.mirror(image.resize((width, 96), Image.Resampling.LANCZOS))
        gray = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(96, width).float() / 255.0
        background = image.filter(ImageFilter.GaussianBlur(radius=5.0))
        background = torch.frombuffer(bytearray(background.tobytes()), dtype=torch.uint8).reshape(96, width).float() / 255.0
        ink = ((background - gray).clamp(min=0) * 1.6).clamp(0, 1)
        target = torch.tensor([self.vocab.get(c, self.vocab["<UNK>"]) for c in row["transcript"]], dtype=torch.long)
        minimum = len(target) + sum(a == b for a, b in zip(row["transcript"], row["transcript"][1:]))
        if self.policy != "originals" and width // 4 < minimum:
            raise ValueError(f"Alignement CTC impossible : {row['file_name']}, vue {view}.")
        return torch.stack((gray * 2 - 1, ink * 2 - 1)), target, row, width


class TensorBudgetSampler:
    """Limit the complete padded N*C*H*W tensor, including both channels."""
    def __init__(self, dataset, batch_size=4, max_elements=1000000, seed=42):
        if batch_size < 1 or max_elements < 192:
            raise ValueError("Budget de microbatch invalide.")
        self.dataset, self.batch_size, self.max_elements, self.seed = dataset, batch_size, max_elements, seed
        self.epoch = 0
        self.widths = [math.ceil(w * 1.2) + 32 for w in dataset.expected_widths]
        if max(self.widths) * 192 > max_elements:
            raise ValueError("Une image seule depasse le budget ; augmenter --max-batch-elements sans tronquer le texte.")

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        bucket_size = self.batch_size * 32
        result = []
        for start in range(0, len(indices), bucket_size):
            bucket = sorted(indices[start:start + bucket_size], key=lambda i: self.widths[i])
            current, width = [], 0
            for i in bucket:
                proposed = max(width, self.widths[i])
                if current and (len(current) >= self.batch_size or (len(current) + 1) * proposed * 192 > self.max_elements):
                    result.append(current)
                    current, width = [], 0
                current.append(i)
                width = max(width, self.widths[i])
            if current:
                result.append(current)
        rng.shuffle(result)
        return result

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())


def collapse_tokens(ids):
    result, previous = [], None
    for value in ids:
        value = int(value)
        if value and value != previous:
            result.append(value)
        previous = value
    return tuple(result)


def training_candidates(log_probs, reference, vocab, beam_width=8, nbest=4, token_top_k=8):
    if not reference or any(c not in vocab for c in reference):
        raise ValueError("MWER exige une reference TRAIN non vide dans le vocabulaire fige.")
    hypotheses = prefix_beam_search(log_probs.detach().cpu().tolist(), beam_width=beam_width, nbest=nbest, token_top_k=token_top_k)
    candidates = [tokens for tokens, _ in hypotheses]
    candidates += [collapse_tokens(log_probs.detach().argmax(-1).tolist()), tuple(vocab[c] for c in reference)]
    # The ground truth is admitted ONLY here, on TRAIN, to give risk zero a candidate.
    return list(dict.fromkeys(candidates))


def minimum_word_risk(log_probs, candidates, reference, vocab, temperature=1.0):
    """Exact differentiable CTC rescoring on a detached, restricted N-best set."""
    if temperature <= 0 or not candidates:
        raise ValueError("Temperature et hypotheses MWER invalides.")
    steps, classes = log_probs.shape
    if any(len(h) + sum(a == b for a, b in zip(h, h[1:])) > steps or any(t <= 0 or t >= classes for t in h) for h in candidates):
        raise ValueError("Hypothese CTC impossible.")
    targets = torch.tensor([t for h in candidates for t in h], dtype=torch.long, device=log_probs.device)
    losses = F.ctc_loss(log_probs[:, None, :].expand(-1, len(candidates), -1).contiguous(), targets,
                        torch.full((len(candidates),), steps, dtype=torch.long), torch.tensor([len(h) for h in candidates]),
                        blank=0, reduction="none", zero_infinity=False)
    if not torch.isfinite(losses).all():
        raise RuntimeError("Probabilite CTC MWER non finie.")
    posterior = (-losses / temperature).softmax(0)
    words = reference.split()
    risks = log_probs.new_tensor([edit_distance(words, tokens_to_text(h, vocab).split()) / max(1, len(words)) for h in candidates])
    return (posterior * risks).sum()


@contextlib.contextmanager
def sam_second_pass(model):
    layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    settings = [m.track_running_stats for m in layers]
    try:
        for layer in layers:
            layer.track_running_stats = False
        yield
    finally:
        for layer, setting in zip(layers, settings):
            layer.track_running_stats = setting


def rng_state(device):
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if device.type == "cuda" else None


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def train_group(model, group, optimizer, vocab, args, device, global_step, epoch):
    examples = sum(len(b["rows"]) for b in group)
    active = args.trial == "D" and global_step % args.mwer_every == 0
    positions = [(i, j) for i, batch in enumerate(group) for j in range(len(batch["rows"]))]
    selected = set(random.Random(args.seed + epoch * 1000003 + global_step).sample(positions, min(args.mwer_samples, examples))) if active else set()
    candidates = {}
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)

    def passes(make_candidates):
        total, ctc_sum, risk_sum = 0.0, 0.0, 0.0
        for i, batch in enumerate(group):
            if batch["images"].numel() > args.max_batch_elements:
                raise ValueError("Microbatch augmente au-dela du budget ; choisir un budget superieur sans tronquer.")
            if any(row.get("split") != "train" for row in batch["rows"]):
                raise ValueError("Une reference hors TRAIN a atteint l'optimisation.")
            count = len(batch["rows"])
            logits = model(batch["images"].to(device, non_blocking=True))
            log_probs = logits.log_softmax(-1)
            ctc = criterion(log_probs.transpose(0, 1), batch["targets"].to(device), batch["widths"] // 4, batch["target_lengths"])
            loss = ctc * (count / examples)
            for j, row in enumerate(batch["rows"]):
                if (i, j) not in selected:
                    continue
                lp = log_probs[j, :int(batch["widths"][j]) // 4]
                if make_candidates:
                    candidates[i, j] = training_candidates(lp, row["transcript"], vocab, args.mwer_beam_width, args.mwer_nbest, args.mwer_token_top_k)
                risk = minimum_word_risk(lp, candidates[i, j], row["transcript"], vocab, args.mwer_temperature)
                loss = loss + args.mwer_weight * risk / len(selected)
                risk_sum += float(risk.detach())
            if not torch.isfinite(loss):
                raise RuntimeError("Perte CTC/MWER non finie.")
            loss.backward()
            total += float(loss.detach())
            ctc_sum += float(ctc.detach()) * count
        return total, ctc_sum, risk_sum

    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = rng_state(device)
    first = passes(True)
    after = rng_state(device)
    second_loss = first[0]
    if TRIALS[args.trial]["sam"]:
        gradients = [p for p in model.parameters() if p.grad is not None]
        norm = torch.stack([p.grad.norm(2) for p in gradients]).norm(2)
        if not torch.isfinite(norm):
            raise RuntimeError("Norme de gradient SAM non finie.")
        originals = []
        try:
            with torch.no_grad():
                for parameter in gradients:
                    originals.append((parameter, parameter.detach().clone()))
                    parameter.add_(parameter.grad * (args.sam_rho / (norm + 1e-12)))
            optimizer.zero_grad(set_to_none=True)
            restore_rng(before)
            with sam_second_pass(model):
                second_loss = passes(False)[0]
        finally:
            with torch.no_grad():
                for parameter, original in originals:
                    parameter.copy_(original)
            restore_rng(after)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(total=first[0], ctc_sum=first[1], risk_sum=first[2], risk_examples=len(selected), examples=examples,
                second_loss=second_loss, gradient_norm=float(gradient_norm))


def train_epoch(model, batches, optimizer, vocab, args, device, epoch, global_step):
    import time
    start = time.monotonic()
    stats = dict(total=0.0, ctc_sum=0.0, risk_sum=0.0, risk_examples=0, examples=0, sam_loss=0.0, updates=0, max_gradient_norm=0.0)
    iterator = iter(batches)
    while group := list(itertools.islice(iterator, args.accumulate)):
        global_step += 1
        values = train_group(model, group, optimizer, vocab, args, device, global_step, epoch)
        n = values["examples"]
        stats["total"] += values["total"] * n
        stats["sam_loss"] += values["second_loss"] * n
        for field in ("ctc_sum", "risk_sum", "risk_examples", "examples"):
            stats[field] += values[field]
        stats["updates"] += 1
        stats["max_gradient_norm"] = max(stats["max_gradient_norm"], values["gradient_norm"])
        if stats["updates"] == 1 or stats["updates"] % args.log_every == 0:
            print(f"V14-{args.trial} | epoch {epoch}/{args.epochs} | update {stats['updates']}/{math.ceil(len(batches)/args.accumulate)} | "
                  f"loss {stats['total']/stats['examples']:.4f} | CTC {stats['ctc_sum']/stats['examples']:.4f} | "
                  f"MWER N={stats['risk_examples']} | {(time.monotonic()-start)/60:.1f} min", flush=True)
    return stats, global_step
