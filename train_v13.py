"""IX-World: V13 semantic supervision, fixed grouped split, GTX 1060 recipe."""
import argparse
import csv
import itertools
import json
import math
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hausa_ocr import AUTHOR, SOURCE_REPOSITORY
from hausa_ocr.data import LineDataset, collate, decode, edit_distance
from hausa_ocr.metrics import UWER_DEFINITION, normalize_uwer
from hausa_ocr.reporting import append_history, cer_trend, print_epoch_report
from hausa_ocr.semantic import (coverage, context_values, digest, fit_context_schema, load_context,
                                require_frozen_split, sample_id, semantic_loss)
from hausa_ocr.v13 import PixelBudgetSampler, SemanticCRNN
from prepare_hausa import sha256, write_json
from train_hausa import load_protocol, save_checkpoint


def device_context(rows, contexts, schema, device):
    return {key: value.to(device) for key, value in context_values(rows, contexts, schema).items()}


def make_loader(dataset, args, training=False):
    sampler = PixelBudgetSampler(dataset, args.batch_size, args.max_batch_pixels, args.seed, training)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=args.device == "cuda"), sampler


def load_semantics(folder, train_rows, contexts, report):
    target_path = folder / "train_targets.pt"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if sha256(target_path) != manifest.get("cache_sha256"):
        raise ValueError("Cache semantique modifie ou preparation incomplete : relancer preparer_v13.cmd dans un nouveau dossier.")
    cached = torch.load(target_path, map_location="cpu", weights_only=True)
    identity = cached["identity"]
    train_contexts = {str(row["source"]): contexts[str(row["source"])] for row in train_rows}
    expected_schema = fit_context_schema(train_rows, contexts)
    if (not cached.get("training_only") or not cached.get("teacher_frozen") or
            identity["split_sha256"] != report["split_sha256"] or identity["vocab_sha256"] != report["vocab_sha256"] or
            identity["train_rows_sha256"] != digest(train_rows) or identity["train_contexts"] != train_contexts or
            cached["context_schema"] != expected_schema or digest(identity) != cached["identity_sha256"] or
            cached["identity_sha256"] != manifest["identity_sha256"]):
        raise ValueError("Cibles semantiques incompatibles avec le TRAIN ou ses metadonnees.")
    if cached["sample_ids"] != [sample_id(row) for row in train_rows]:
        raise ValueError("Le cache semantique ne contient pas exactement les identifiants TRAIN dans leur ordre archive.")
    if len(cached.get("positive_keys", [])) != len(train_rows) or any(not isinstance(key, str) or not key for key in cached["positive_keys"]):
        raise ValueError("Identites des contextes textuels manquantes dans le cache.")
    annotations = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
    if digest(annotations) != identity["annotations_sha256"]:
        raise ValueError("Annotations semantiques modifiees depuis la preparation.")
    targets = cached["targets"]
    if targets.dtype != torch.float32 or targets.shape != (len(train_rows), 768) or not torch.isfinite(targets).all():
        raise ValueError("Cibles semantiques invalides.")
    known = torch.tensor([contexts[str(row["source"])]["language"] is not None for row in train_rows])
    if bool(known.any()) and not torch.allclose(targets[known].norm(dim=-1), torch.ones(int(known.sum())), atol=1e-4):
        raise ValueError("Les cibles semantiques des langues connues doivent etre normalisees et non nulles.")
    return cached, {key: index for index, key in enumerate(cached["sample_ids"])}, sha256(target_path)


@torch.inference_mode()
def evaluate(model, batches, vocab, contexts, schema, device):
    model.eval()
    counts = {key: 0 for key in ("character_errors", "reference_characters", "word_errors", "reference_words", "normalized_word_errors", "normalized_reference_words", "exact", "examples")}
    predictions = []
    for batch in batches:
        logits = model(batch["images"].to(device), batch["widths"], device_context(batch["rows"], contexts, schema, device))
        ids = logits.argmax(-1).cpu()
        for index, row in enumerate(batch["rows"]):
            prediction = decode(ids[index, :int(batch["widths"][index]) // 4], vocab)
            reference = row["transcript"]
            normalized_reference = normalize_uwer(reference).split()
            values = {"character_errors": edit_distance(reference, prediction), "reference_characters": len(reference),
                      "word_errors": edit_distance(reference.split(), prediction.split()), "reference_words": len(reference.split()),
                      "normalized_word_errors": edit_distance(normalized_reference, normalize_uwer(prediction).split()),
                      "normalized_reference_words": len(normalized_reference), "exact": int(reference == prediction), "examples": 1}
            for key, value in values.items():
                counts[key] += value
            ctx = contexts[str(row["source"])]
            predictions.append(dict(sample_id=sample_id(row), file_name=row["file_name"], public_origin=row["partition"], source=row["source"],
                                    language=ctx["language"], locality=ctx["locality"], year_start=ctx["year_start"], year_end=ctx["year_end"],
                                    reference=reference, prediction=prediction, character_errors=values["character_errors"],
                                    word_errors=values["word_errors"], normalized_word_errors=values["normalized_word_errors"]))
    counts.update(cer=counts["character_errors"] / max(1, counts["reference_characters"]),
                  wer=counts["word_errors"] / max(1, counts["reference_words"]),
                  uwer=counts["normalized_word_errors"] / max(1, counts["normalized_reference_words"]),
                  exact_rate=counts["exact"] / max(1, counts["examples"]))
    return counts, predictions


def semantic_weight(args, epoch):
    return args.semantic_weight * min(1.0, max(0.0, (epoch - args.semantic_start_epoch + 1) / args.semantic_warmup))


def train_epoch(model, batches, optimizer, criterion, targets, positive_keys, target_index, contexts, schema, args, epoch, device):
    model.train()
    stats = dict(total=0.0, ctc=0.0, semantic=0.0, cosine=0.0, contrastive=0.0, examples=0, optimizer_steps=0)
    weight = semantic_weight(args, epoch)
    iterator, processed, beginning = iter(batches), 0, time.monotonic()
    while True:
        group = list(itertools.islice(iterator, args.accumulate))
        if not group:
            break
        total_examples = sum(len(batch["rows"]) for batch in group)
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            count = len(batch["rows"])
            if batch["images"].numel() > args.max_batch_pixels:
                raise ValueError("Batch augmente au-dela du budget de pixels ; augmenter le budget ou reduire --batch-size.")
            context = device_context(batch["rows"], contexts, schema, device)
            logits, visual = model(batch["images"].to(device), batch["widths"], context, return_semantic=True)
            ctc = criterion(logits.log_softmax(-1).transpose(0, 1), batch["targets"].to(device), batch["widths"] // 4, batch["target_lengths"])
            indices = [target_index[sample_id(row)] for row in batch["rows"]]
            teacher = targets[indices].to(device)
            semantic, cosine, contrastive = semantic_loss(visual, teacher, [row["transcript"] for row in batch["rows"]], context["language"],
                                                          positive_keys=[positive_keys[i] for i in indices])
            loss = ctc + weight * semantic
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie : verifier CTC et cibles semantiques.")
            (loss * count / total_examples).backward()
            for key, value in (("total", loss), ("ctc", ctc), ("semantic", semantic), ("cosine", cosine), ("contrastive", contrastive)):
                stats[key] += float(value.detach()) * count
            stats["examples"] += count
            processed += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        stats["optimizer_steps"] += 1
        if processed <= args.accumulate or processed // args.log_every != (processed - len(group)) // args.log_every or processed == len(batches):
            seconds = time.monotonic() - beginning
            remaining = seconds / processed * (len(batches) - processed) / 60
            print(f"v13 | epoch {epoch}/{args.epochs} | microbatch {processed}/{len(batches)} | loss {stats['total']/stats['examples']:.4f} | CTC {stats['ctc']/stats['examples']:.4f} | sem {stats['semantic']/stats['examples']:.4f} | lambda {weight:.4f} | reste ~{remaining:.1f} min", flush=True)
    for key in ("total", "ctc", "semantic", "cosine", "contrastive"):
        stats[key] /= stats["examples"]
    return stats


def snapshot_source(output):
    root, destination = Path(__file__).resolve().parent, output / "source"
    shutil.copytree(root / "hausa_ocr", destination / "hausa_ocr", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train_v13.py", "prepare_v13_semantics.py", "train_hausa.py", "prepare_hausa.py", "requirements-v13-semantic.txt", "README_V13.md"):
        shutil.copy2(root / name, destination / name)
    return {str(path.relative_to(destination)).replace("\\", "/"): sha256(path) for path in sorted(destination.rglob("*.py"))}


def write_run_card(output, config, summary):
    lines = ["# V13 Hausa Ajami — IX-World", "", "CNN 6 convolutions / GroupNorm / BiLSTM 2 couches / CTC, avec supervision auxiliaire AfriBERTa figee.",
             "", f"Split commun : {config['counts']}. Checkpoint retenu : minimum CER validation, epoque {summary['selected_epoch']}.", "",
             "| Partition | Segments | CER | WER | UWER | Exact |", "|---|---:|---:|---:|---:|---:|"]
    for part, result in summary["partitions"].items():
        lines.append(f"| {part} | {result['examples']} | {result['cer']:.4%} | {result['wer']:.4%} | {result['uwer']:.4%} | {result['exact_rate']:.4%} |")
    lines += ["", "## Supervision et limites", "", f"Encodeur : {config['teacher_repo']} @ {config['teacher_revision']}. Cibles issues du TRAIN uniquement.",
              f"Couverture du contexte TRAIN : `{config['context_coverage']['train']}`. Contextes textuels verifies : {config['verified_contextual_annotations']}.",
              "Les valeurs d'epoque/localite inconnues sont masquees. Un encodeur contemporain et des mots isoles ne demontrent pas une comprehension historique ou regionale.",
              "Aucune detection automatique de langue, correction libre, traduction ou adaptation continue non supervisee n'est effectuee.",
              "Le modele transcrit en alphabet latin ; il ne remplace pas les formes anciennes par des formes modernes.",
              "L'encodeur n'est pas necessaire a l'inference. Aucun gain sur V10/V11 n'est garanti ; comparer les resultats sur le meme split.",
              "", "## Reproductibilite", "", "Voir config.json, split.json, vocab.json, context.json, semantic_manifest.json et source/.",
              "history.csv documente chaque epoque ; les scores train/test sont mesures a la fin sur les images originales.",
              f"UWER : {UWER_DEFINITION}", "", "Source : https://github.com/IX-World/OCR-AJAMI-Hausa", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description="V13 IX-World : supervision semantique contextuelle, meme split, GTX 1060 6 Go.")
    base = Path(__file__).resolve().parent.parent
    parser.add_argument("--dataset", type=Path, default=base / "Hausa_repo_2026-10-15")
    parser.add_argument("--protocol", type=Path, default=base / "protocol_hausa_all_grouped_20261016")
    default_context = base / "v13_context.json"
    parser.add_argument("--context", type=Path, default=default_context if default_context.exists() else Path(__file__).with_name("configs") / "v13_context.json")
    parser.add_argument("--semantics", type=Path, default=base / "v13_semantics")
    parser.add_argument("--output", type=Path, default=base / "runs_hausa_all_grouped_20261016" / "v13_seed42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--max-batch-pixels", type=int, default=600000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--semantic-weight", type=float, default=0.1)
    parser.add_argument("--semantic-start-epoch", type=int, default=4)
    parser.add_argument("--semantic-warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reprendre uniquement last.pt de CE run V13 incomplet.")
    parser.add_argument("--evaluate-only", action="store_true", help="Reevaluer le checkpoint V13 deja entraine sur les trois partitions.")
    parser.add_argument("--check-only", action="store_true", help="Verifier chemins et une inference de validation, sans apprendre ni sauvegarder de poids.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--show-examples", type=int, default=5)
    args = parser.parse_args()
    if sum((args.resume, args.evaluate_only, args.check_only)) > 1 or args.fresh_start and (args.resume or args.evaluate_only):
        parser.error("Choisir un lancement neuf, une reprise V13 ou une evaluation.")
    if min(args.epochs, args.min_epochs, args.patience, args.batch_size, args.accumulate, args.max_batch_pixels, args.threads, args.log_every, args.semantic_start_epoch, args.semantic_warmup) < 1 or args.min_epochs > args.epochs:
        parser.error("Parametres entiers positifs, min-epochs <= epochs.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.semantic_weight < 0 or args.show_examples < 0:
        parser.error("Parametres d'apprentissage invalides.")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible. Utiliser PyTorch 2.5.1+cu121 compatible GTX 1060, ou --device cpu pour un diagnostic.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    print("Verification du split commun et des empreintes des images...", flush=True)
    splits, vocab, report = load_protocol(args.dataset, args.protocol)
    require_frozen_split(report)
    raw_context, contexts = load_context(args.context, set(report["source_to_group"]))
    cached, target_index, cache_hash = load_semantics(args.semantics, splits["train"], contexts, report)
    schema = cached["context_schema"]
    model = SemanticCRNN(len(vocab), schema).to(device)
    train_data = LineDataset(args.dataset, splits["train"], vocab, "v10", training=True, seed=args.seed)
    train_batches, sampler = make_loader(train_data, args, training=True)
    val_data = LineDataset(args.dataset, splits["validation"], vocab, "v10", training=False)
    val_batches, _ = make_loader(val_data, args)
    print(f"V13 | TRAIN 4972 / VAL 200 / TEST 401 | {len(train_data)} vues train/epoque (8/image)", flush=True)
    print(f"Microbatch <= {args.batch_size}, budget {args.max_batch_pixels} pixels, accumulation {args.accumulate}, batch nominal <= {args.batch_size*args.accumulate}", flush=True)
    print(f"Device : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | Precision float32 | Encodeur absent du GPU", flush=True)
    context_coverage = {part: coverage(rows, contexts) for part, rows in splits.items()}
    print(f"Contexte TRAIN : {context_coverage['train']} | annotations textuelles verifiees {cached['verified_contextual_annotations']}", flush=True)
    if context_coverage["train"]["known_period"] == 0 or context_coverage["train"]["known_locality"] == 0:
        print("Epoque/localite non documentees : branches correspondantes masquees, aucune capacite historique/regionale affirmee.", flush=True)
    if args.check_only:
        batch = next(iter(val_batches))
        with torch.inference_mode():
            model.eval()
            logits, vector = model(batch["images"].to(device), batch["widths"], device_context(batch["rows"], contexts, schema, device), True)
        print(f"Verification terminee : logits {tuple(logits.shape)}, semantique {tuple(vector.shape)}. Aucun poids entraine sauvegarde.", flush=True)
        return
    output = args.output.resolve()
    if not (args.resume or args.evaluate_only) and output.exists() and any(output.iterdir()):
        raise FileExistsError("Sortie deja utilisee : choisir un autre --output pour repartir de zero.")
    if args.resume and (output / "metrics.json").exists():
        raise ValueError("Ce run est termine et son test deja mesure. La reprise est reservee aux runs V13 interrompus.")
    git = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True)
    config = dict(author=AUTHOR, source_repository=SOURCE_REPOSITORY, source_commit=git.stdout.strip() if git.returncode == 0 else None,
                  version="v13", dataset=str(args.dataset.resolve()), dataset_repo=report["dataset_repo"], dataset_revision=report["dataset_revision"],
                  protocol_id=report["protocol_id"], split_sha256=report["split_sha256"], vocab_sha256=report["vocab_sha256"], counts=report["counts"],
                  height=96, channels=1, rtl_mirror=True, padding={"grayscale": 1.0}, classes=len(vocab),
                  cnn_convolutions=6, normalization="GroupNorm_8_groups", cnn_dropout=0.1, lstm_hidden_per_direction=256, lstm_layers=2,
                  lstm_dropout=0.2, sequence_dropout=0.2, packed_lstm=True, semantic_dim=768, context_schema=schema, context_coverage=context_coverage,
                  context_sha256=digest(raw_context), resolved_context_sha256=digest(contexts), teacher_repo=cached["identity"]["teacher_repo"],
                  teacher_revision=cached["identity"]["teacher_revision"], teacher_frozen=True, teacher_device="cpu_preparation_only",
                  semantic_identity_sha256=cached["identity_sha256"], semantic_cache_sha256=cache_hash, verified_contextual_annotations=cached["verified_contextual_annotations"],
                  semantic_weight=args.semantic_weight, semantic_start_epoch=args.semantic_start_epoch, semantic_warmup=args.semantic_warmup,
                  semantic_loss="0.5_cosine_plus_0.5_same_language_multi_positive_contrastive_temperature_0.1", semantic_negatives="actual_microbatch_only",
                  parameter_count=sum(p.numel() for p in model.parameters()), augmentation="fixed_8_views", views_per_image=8, virtual_train_size=len(train_data),
                  optimizer="AdamW", learning_rate=args.learning_rate, weight_decay=args.weight_decay, gradient_clip=5.0,
                  batch_size=args.batch_size, accumulate=args.accumulate, max_batch_pixels=args.max_batch_pixels, max_epochs=args.epochs,
                  min_epochs=args.min_epochs, patience=args.patience, seed=args.seed, from_scratch=True, ocr_checkpoint_initialization=None,
                  precision="float32", python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, uwer_definition=UWER_DEFINITION,
                  final_evaluation="train_validation_test_originals_using_minimum_validation_CER_checkpoint", automatic_language_detection=False,
                  online_self_training=False, scheduler={"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 3, "min_lr": 1e-6})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    criterion = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    records, start_epoch, best_cer, patience_best, without_gain, elapsed_total, peak_total = [], 1, float("inf"), float("inf"), 0, 0.0, 0.0
    if args.resume or args.evaluate_only:
        checkpoint = torch.load(output / ("last.pt" if args.resume else "best_cer.pt"), map_location="cpu", weights_only=False)
        if checkpoint["vocab"] != vocab:
            raise ValueError("Vocabulaire incompatible.")
        ignored = {"source_commit", "max_epochs", "gpu", "cuda", "python", "torch"}
        for key in set(config) - ignored:
            if checkpoint["config"].get(key) != config[key]:
                raise ValueError(f"Configuration V13 incompatible : {key}. Conserver les arguments du lancement initial.")
        for name, expected in checkpoint["config"]["source_sha256"].items():
            current = Path(__file__).resolve().parent / name
            if not current.exists() or sha256(current) != expected:
                raise ValueError(f"Source modifiee depuis le run V13 : {name}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        config = dict(checkpoint["config"], max_epochs=args.epochs if args.resume else checkpoint["config"]["max_epochs"])
        elapsed_total, peak_total = checkpoint["elapsed_seconds"], checkpoint["peak_vram_mib"]
        if (output / "history.csv").exists():
            with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
        if args.resume:
            if not records or int(records[-1]["epoch"]) != checkpoint["epoch"]:
                raise ValueError("Historique et dernier checkpoint desynchronises ; conserver le run pour verification.")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch, best_cer = checkpoint["epoch"] + 1, checkpoint["best_cer"]
            patience_best, without_gain = checkpoint["patience_best"], checkpoint["without_gain"]
            random.setstate(checkpoint["python_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        output.mkdir(parents=True, exist_ok=True)
        config["source_sha256"] = snapshot_source(output)
        for name in ("split.json", "vocab.json", "protocol.json", "quarantine.json", "source_groups.json", "excluded_public_train.json"):
            shutil.copy2(args.protocol / name, output / name)
        write_json(output / "context.json", raw_context)
        shutil.copy2(args.semantics / "manifest.json", output / "semantic_manifest.json")
        shutil.copy2(args.semantics / "annotations.json", output / "semantic_annotations.json")
    write_json(output / "config.json", config)
    try:
        if not args.evaluate_only:
            for epoch in range(start_epoch, args.epochs + 1):
                train_data.epoch = sampler.epoch = epoch
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                beginning, lr_used = time.monotonic(), optimizer.param_groups[0]["lr"]
                print(f"\nDEBUT EPOCH {epoch}/{args.epochs} | {len(train_batches)} microbatches | ~{math.ceil(len(train_batches)/args.accumulate)} mises a jour", flush=True)
                stats = train_epoch(model, train_batches, optimizer, criterion, cached["targets"], cached["positive_keys"], target_index, contexts, schema, args, epoch, device)
                val, predictions = evaluate(model, val_batches, vocab, contexts, schema, device)
                scheduler.step(val["cer"])
                seconds = time.monotonic() - beginning
                elapsed_total += seconds
                epoch_peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
                allocated = torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
                peak_total = max(peak_total, epoch_peak)
                improved = val["cer"] < best_cer
                best_cer = min(best_cer, val["cer"])
                if val["cer"] < patience_best - 0.0005:
                    patience_best, without_gain = val["cer"], 0
                else:
                    without_gain += 1
                record = dict(epoch=epoch, train_loss=stats["total"], train_ctc_loss=stats["ctc"], train_semantic_loss=stats["semantic"],
                              train_cosine_loss=stats["cosine"], train_contrastive_loss=stats["contrastive"], semantic_weight=semantic_weight(args, epoch),
                              optimizer_steps=stats["optimizer_steps"], val_cer=val["cer"], val_wer=val["wer"], val_uwer=val["uwer"], val_exact_rate=val["exact_rate"],
                              lr_used=lr_used, lr=optimizer.param_groups[0]["lr"], seconds=seconds, vram_allocated_mib=allocated,
                              epoch_peak_vram_mib=epoch_peak, peak_vram_mib=peak_total)
                slope, gain = cer_trend(records + [record])
                record.update(val_cer_slope=slope, val_cer_window_gain=gain)
                append_history(output / "history.csv", records, record)
                saved = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                             config=config, vocab=vocab, epoch=epoch, validation=val, best_cer=best_cer, patience_best=patience_best,
                             without_gain=without_gain, elapsed_seconds=elapsed_total, peak_vram_mib=peak_total, python_rng=random.getstate(),
                             torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
                if improved:
                    save_checkpoint(output / "best_cer.pt", saved)
                save_checkpoint(output / "last.pt", saved)
                print_epoch_report(config, record, records, predictions, without_gain, args.show_examples)
                print(f"CTC {stats['ctc']:.4f} | semantique {stats['semantic']:.4f} | lambda {record['semantic_weight']:.4f}", flush=True)
                if epoch >= args.min_epochs and without_gain >= args.patience:
                    print("Arret anticipe sur le CER de validation.", flush=True)
                    break
    except KeyboardInterrupt:
        print("Interruption : le dernier checkpoint d'epoque complete est conserve. Reprendre avec reprendre_v13.cmd.", flush=True)
        raise SystemExit(130) from None
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("VRAM insuffisante. Liberer la carte ou lancer dans un nouveau --output avec --batch-size 4 --accumulate 6 --max-batch-pixels 400000.") from None
    checkpoint = torch.load(output / "best_cer.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_hash = sha256(output / "best_cer.pt")
    summary = dict(version="v13", selected_epoch=checkpoint["epoch"], checkpoint_sha256=checkpoint_hash, split_sha256=report["split_sha256"],
                   dataset_revision=report["dataset_revision"], selection="minimum_validation_CER", uwer_definition=UWER_DEFINITION,
                   elapsed_training_seconds=elapsed_total, peak_vram_mib=peak_total, evaluated_at_utc=datetime.now(timezone.utc).isoformat(), partitions={})
    for part, rows in splits.items():
        print(f"Evaluation finale {part.upper()} : {len(rows)} images originales sans augmentation...", flush=True)
        dataset = LineDataset(args.dataset, rows, vocab, "v10", training=False)
        batches, _ = make_loader(dataset, args)
        metrics, predictions = evaluate(model, batches, vocab, contexts, schema, device)
        metrics.update(version="v13", evaluated_partition=part, selected_epoch=checkpoint["epoch"], checkpoint_sha256=checkpoint_hash,
                       split_sha256=report["split_sha256"], dataset_revision=report["dataset_revision"], augmented=False, uwer_definition=UWER_DEFINITION)
        summary["partitions"][part] = metrics
        write_json(output / f"{part}_metrics.json", metrics)
        for row in predictions:
            row["evaluated_partition"] = part
        with (output / f"{part}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    write_json(output / "metrics.json", summary)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["partition", "examples", "cer", "wer", "uwer", "exact_rate", "character_errors", "reference_characters", "word_errors", "reference_words", "normalized_word_errors", "normalized_reference_words"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(values, partition=part) for part, values in summary["partitions"].items())
    write_run_card(output, config, summary)
    print(f"\nBILAN FINAL V13 | meilleur CER validation @ {checkpoint['epoch']}")
    print(f"{'Partition':12s} {'Segments':>8s} {'CER':>10s} {'WER':>10s} {'UWER':>10s} {'Exact':>10s}")
    for part, values in summary["partitions"].items():
        print(f"{part:12s} {values['examples']:8d} {values['cer']:10.2%} {values['wer']:10.2%} {values['uwer']:10.2%} {values['exact_rate']:10.2%}")
    print(f"Resultats et README : {output}", flush=True)


if __name__ == "__main__":
    main()
