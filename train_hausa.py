import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hausa_ocr import AUTHOR, SOURCE_REPOSITORY
from hausa_ocr.data import LineDataset, WidthSampler, collate, decode, edit_distance
from hausa_ocr.models import build_model
from prepare_hausa import sha256, write_json


def load_protocol(dataset, protocol):
    report = json.loads((protocol / "protocol.json").read_text(encoding="utf-8"))
    for name, field in (("split.json", "split_sha256"), ("vocab.json", "vocab_sha256")):
        if sha256(protocol / name) != report[field]:
            raise ValueError(f"Le fichier {name} ne correspond plus au protocole.")
    for part, field in (("train", "corrected_train_metadata_sha256"), ("test", "test_metadata_sha256")):
        if sha256(dataset / "data" / part / "metadata.jsonl") != report[field]:
            raise ValueError(f"Metadonnees {part} modifiees depuis la preparation.")
    splits = json.loads((protocol / "split.json").read_text(encoding="utf-8"))
    vocab = json.loads((protocol / "vocab.json").read_text(encoding="utf-8"))
    for part, rows in splits.items():
        if len(rows) != report["counts"][part]:
            raise ValueError("Effectif du split modifie.")
        for row in rows:
            path = dataset / "data" / row["partition"] / row["file_name"]
            if sha256(path) != row["file_sha256"]:
                raise ValueError(f"Image modifiee depuis la preparation : {path}")
    return splits, vocab, report


@torch.inference_mode()
def evaluate(model, loader, vocab, device):
    model.eval()
    predictions = []
    counts = {"character_errors": 0, "reference_characters": 0, "word_errors": 0, "reference_words": 0, "exact": 0, "examples": 0}
    for batch in loader:
        logits = model(batch["images"].to(device, non_blocking=True))
        ids = logits.argmax(-1).cpu()
        for i, row in enumerate(batch["rows"]):
            prediction = decode(ids[i, :int(batch["widths"][i]) // 4], vocab)
            reference = row["transcript"]
            char_errors = edit_distance(reference, prediction)
            word_errors = edit_distance(reference.split(), prediction.split())
            counts["character_errors"] += char_errors
            counts["reference_characters"] += len(reference)
            counts["word_errors"] += word_errors
            counts["reference_words"] += len(reference.split())
            counts["exact"] += int(reference == prediction)
            counts["examples"] += 1
            predictions.append({"partition": row["partition"], "file_name": row["file_name"], "source": row["source"], "reference": reference, "prediction": prediction, "character_errors": char_errors, "word_errors": word_errors})
    counts.update(cer=counts["character_errors"] / max(1, counts["reference_characters"]), wer=counts["word_errors"] / max(1, counts["reference_words"]), exact_rate=counts["exact"] / max(1, counts["examples"]))
    return counts, predictions


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def loader(dataset, batch_size, device, sampler=None):
    common = dict(dataset=dataset, num_workers=0, collate_fn=collate, pin_memory=device.type == "cuda")
    if sampler is not None:
        return DataLoader(batch_sampler=sampler, **common)
    return DataLoader(batch_size=batch_size, shuffle=False, **common)


def main():
    parser = argparse.ArgumentParser(description="IX-World : V9--V12 sur le meme test public Hausa")
    base = Path(__file__).resolve().parent.parent
    parser.add_argument("--version", choices=("v9", "v10", "v11", "v12"), required=True)
    parser.add_argument("--dataset", type=Path, default=base / "Hausa_repo_2026-10-15")
    parser.add_argument("--protocol", type=Path, default=base / "protocol_hausa_20261015")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    output = (args.output or base / "runs_hausa_20261015" / f"{args.version}_seed{args.seed if args.seed is not None else 42}").resolve()
    previous_config = {}
    if (args.resume or args.test_only) and (output / "config.json").exists():
        previous_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for name, default in {"epochs": 100, "min_epochs": 30, "patience": 15, "seed": 42, "batch_size": 20 if args.version == "v12" else 24}.items():
        if getattr(args, name) is None:
            field = "max_epochs" if name == "epochs" else name
            setattr(args, name, previous_config.get(field, default))
    if min(args.epochs, args.patience, args.log_every) < 1 or args.min_epochs < 1 or args.min_epochs > args.epochs:
        parser.error("Epoques, patience et frequence doivent etre positives ; min-epochs <= epochs.")
    if args.batch_size < 1:
        parser.error("batch-size doit etre positif")
    args.dataset, args.protocol = args.dataset.resolve(), args.protocol.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible. Verifier PyTorch/CUDA ou preciser --device cpu.")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    print("Verification du protocole et des images...", flush=True)
    splits, vocab, report = load_protocol(args.dataset, args.protocol)
    model = build_model(args.version, len(vocab)).to(device)
    datasets = {part: LineDataset(args.dataset, rows, vocab, args.version, part == "train", args.seed) for part, rows in splits.items()}
    sampler = WidthSampler(datasets["train"], args.batch_size, args.seed)
    loaders = {part: loader(data, args.batch_size, device, sampler if part == "train" else None) for part, data in datasets.items()}
    print(f"{args.version.upper()} | TRAIN {len(splits['train'])} | VAL {len(splits['validation'])} | TEST {len(splits['test'])}", flush=True)
    print(f"Vues train/epoque : {len(datasets['train'])} | batches : {len(loaders['train'])} | classes : {len(vocab)}", flush=True)
    print(f"Device : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | Sortie : {output}", flush=True)
    if args.check_only:
        batch = next(iter(loaders["validation"]))
        with torch.inference_mode():
            model.eval()
            result = model(batch["images"][:1].to(device))
        assert result.shape[1] == batch["images"].shape[-1] // 4
        print(f"Verification terminee : logits {tuple(result.shape)}, aucun poids entraine sauvegarde.", flush=True)
        return
    if output.exists() and (output / "config.json").exists() and not (args.resume or args.test_only):
        raise FileExistsError(f"Un run existe deja : {output}. Utiliser --resume ou un autre --output.")
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("--resume exige last.pt dans le dossier de sortie.")
    if args.test_only and not (output / "best_cer.pt").is_file():
        raise FileNotFoundError("--test-only exige best_cer.pt, issu d'un entrainement termine.")
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    git = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True)
    config = {
        "author": AUTHOR, "source_repository": SOURCE_REPOSITORY, "source_commit": git.stdout.strip() if git.returncode == 0 else None,
        "version": args.version, "dataset": str(args.dataset), "dataset_repo": report["dataset_repo"], "dataset_revision": report["dataset_revision"],
        "split_sha256": report["split_sha256"], "vocab_sha256": report["vocab_sha256"], "counts": report["counts"],
        "vocab_policy": report["vocab_policy"], "unseen_characters": report["unseen_characters"],
        "height": 96, "rtl_mirror": True, "channels": 2 if args.version in ("v11", "v12") else 1,
        "ink_blur_radius": 5.0, "ink_gain": 1.6, "padding": {"grayscale": 1.0, "ink": -1.0},
        "cnn_convolutions": 9 if args.version == "v12" else 6, "cnn_dropout": 0.05 if args.version == "v12" else 0.0,
        "lstm_hidden_per_direction": 256, "lstm_layers": 2, "lstm_dropout": 0.2,
        "parameter_count": sum(p.numel() for p in model.parameters()), "classes": len(vocab),
        "augmentation": "dynamic_probability_0.30" if args.version == "v9" else "fixed_8_views",
        "views_per_image": datasets["train"].views, "virtual_train_size": len(datasets["train"]),
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "gradient_clip": 5.0,
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 3, "min_lr": 1e-6},
        "batch_size": args.batch_size, "bucket_size": sampler.bucket_size, "max_epochs": args.epochs,
        "min_epochs": args.min_epochs, "patience": args.patience, "min_delta": 0.0005,
        "seed": args.seed, "from_scratch": True, "precision": "float32", "strict_determinism": False,
        "python": platform.python_version(), "torch": str(torch.__version__), "pillow": importlib.metadata.version("Pillow"),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2 if device.type == "cuda" else None,
        "historical_recipe_changes": ["new_public_test_400", "train_only_vocabulary_with_UNK", "common_white_gray_padding", "common_patience_schedule", "bucket_size_8_batches"],
    }
    start_epoch, best_cer, patience_best, without_gain, elapsed_total, peak_total = 1, float("inf"), float("inf"), 0, 0.0, 0.0
    if args.resume or args.test_only:
        checkpoint = torch.load(output / ("best_cer.pt" if args.test_only else "last.pt"), map_location="cpu", weights_only=False)
        for field in ("version", "split_sha256", "vocab_sha256", "seed", "batch_size", "height", "padding", "min_epochs", "patience"):
            if checkpoint["config"][field] != config[field]:
                raise ValueError(f"Configuration incompatible avec le checkpoint : {field}")
        if checkpoint["vocab"] != vocab:
            raise ValueError("Vocabulaire du checkpoint different.")
        model.load_state_dict(checkpoint["model_state_dict"])
        config = dict(checkpoint["config"], max_epochs=args.epochs if args.resume else checkpoint["config"]["max_epochs"])
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch, best_cer = checkpoint["epoch"] + 1, checkpoint["best_cer"]
            patience_best, without_gain = checkpoint["patience_best"], checkpoint["without_gain"]
            elapsed_total, peak_total = checkpoint["elapsed_seconds"], checkpoint["peak_vram_mib"]
            random.setstate(checkpoint["python_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        for name in ("split.json", "vocab.json", "protocol.json", "quarantine.json"):
            shutil.copy2(args.protocol / name, output / name)
        snapshot = output / "source"
        shutil.copytree(Path(__file__).resolve().parent / "hausa_ocr", snapshot / "hausa_ocr", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("train_hausa.py", "prepare_hausa.py", "publish_hausa.py", "download_hausa_latest.py", "requirements-hausa.txt"):
            shutil.copy2(Path(__file__).resolve().with_name(name), snapshot / name)
        config["source_sha256"] = {str(p.relative_to(snapshot)).replace("\\", "/"): sha256(p) for p in sorted(snapshot.rglob("*.py"))}
    write_json(output / "config.json", config)
    interrupted = False
    try:
        if not args.test_only:
            for epoch in range(start_epoch, args.epochs + 1):
                datasets["train"].epoch = sampler.epoch = epoch
                model.train()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                beginning, running = time.monotonic(), 0.0
                for index, batch in enumerate(loaders["train"], 1):
                    images = batch["images"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits.log_softmax(-1).permute(1, 0, 2), batch["targets"].to(device), batch["widths"] // 4, batch["target_lengths"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Perte non finie : verifier les longueurs et annotations CTC.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    running += float(loss.detach())
                    if index == 1 or index % args.log_every == 0 or index == len(loaders["train"]):
                        seconds = time.monotonic() - beginning
                        remaining = seconds / index * (len(loaders["train"]) - index) / 60
                        print(f"{args.version} | epoch {epoch}/{args.epochs} | batch {index}/{len(loaders['train'])} | loss {running/index:.4f} | reste epoch ~{remaining:.1f} min", flush=True)
                val, _ = evaluate(model, loaders["validation"], vocab, device)
                scheduler.step(val["cer"])
                seconds = time.monotonic() - beginning
                elapsed_total += seconds
                peak_total = max(peak_total, torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0)
                improved = val["cer"] < best_cer
                best_cer = min(best_cer, val["cer"])
                if val["cer"] < patience_best - 0.0005:
                    patience_best, without_gain = val["cer"], 0
                else:
                    without_gain += 1
                record = {"epoch": epoch, "train_loss": running / len(loaders["train"]), "val_cer": val["cer"], "val_wer": val["wer"], "val_exact_rate": val["exact_rate"], "lr": optimizer.param_groups[0]["lr"], "seconds": seconds, "peak_vram_mib": peak_total}
                history = output / "history.csv"
                with history.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(record)
                saved = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "vocab": vocab, "config": config, "epoch": epoch, "validation": val, "best_cer": best_cer, "patience_best": patience_best, "without_gain": without_gain, "elapsed_seconds": elapsed_total, "peak_vram_mib": peak_total, "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
                if improved:
                    save_checkpoint(output / "best_cer.pt", saved)
                save_checkpoint(output / "last.pt", saved)
                print(f"VAL CER {val['cer']:.4%} | WER {val['wer']:.4%} | best CER {best_cer:.4%} | patience {without_gain}/{args.patience} | VRAM {peak_total:.0f} MiB", flush=True)
                if epoch >= args.min_epochs and without_gain >= args.patience:
                    print("Arret anticipe sur la validation.", flush=True)
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("Interruption : le dernier checkpoint complet reste disponible avec --resume.", flush=True)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("VRAM insuffisante. Choisir un nouveau --output et reduire --batch-size (par exemple 12). Le split reste identique.") from None
    if interrupted:
        raise SystemExit(130)
    checkpoint = torch.load(output / "best_cer.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    results, predictions = evaluate(model, loaders["test"], vocab, device)
    results.update(version=args.version, selected_epoch=checkpoint["epoch"], selection="minimum_validation_CER", validation_at_selected_epoch=checkpoint["validation"], checkpoint_sha256=sha256(output / "best_cer.pt"), split_sha256=report["split_sha256"], dataset_revision=report["dataset_revision"], elapsed_training_seconds=elapsed_total if not args.test_only else json.loads((output / "test_metrics.json").read_text(encoding="utf-8")).get("elapsed_training_seconds") if (output / "test_metrics.json").exists() else checkpoint["elapsed_seconds"], peak_vram_mib=peak_total or checkpoint["peak_vram_mib"])
    write_json(output / "test_metrics.json", results)
    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    from publish_hausa import write_model_card
    write_model_card(output)
    print(f"TEST PUBLIC 400 | CER {results['cer']:.4%} | WER {results['wer']:.4%} | exact {results['exact_rate']:.4%}\nResultats : {output}", flush=True)


if __name__ == "__main__":
    main()
