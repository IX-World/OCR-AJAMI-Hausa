"""IX-World: four controlled fine-tuning trials from the same V12 weights."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hausa_ocr import AUTHOR, SOURCE_REPOSITORY
from hausa_ocr.data import collate
from hausa_ocr.metrics import UWER_DEFINITION
from hausa_ocr.models import build_model
from hausa_ocr.reporting import append_history, cer_trend
from hausa_ocr.v14 import COUNTS, TRIALS, V12_CHECKPOINT_SHA256, TrialDataset, TensorBudgetSampler, require_protocol, train_epoch
from prepare_hausa import sha256, write_json
from train_hausa import evaluate, load_protocol, save_checkpoint


DEFAULTS = dict(epochs=30, min_epochs=10, patience=10, batch_size=4, accumulate=5, max_batch_elements=600000,
                eval_batch_size=20, learning_rate=0.0001, weight_decay=0.0001, sam_rho=0.05, seed=42,
                mwer_weight=0.05, mwer_every=4, mwer_samples=1, mwer_beam_width=8, mwer_nbest=4,
                mwer_token_top_k=8, mwer_temperature=1.0)
SOURCE_FILES = ("train_v14.py", "train_hausa.py", "prepare_hausa.py", "hausa_ocr/__init__.py", "hausa_ocr/data.py",
                "hausa_ocr/models.py", "hausa_ocr/augmentations.py", "hausa_ocr/ctc_beam.py", "hausa_ocr/metrics.py",
                "hausa_ocr/protocol.py", "hausa_ocr/reporting.py", "hausa_ocr/v14.py")
SELECTION = "minimum_validation_WER_subject_to_CER_not_above_initial_V12"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_v12(path, vocab, report):
    if sha256(path) != V12_CHECKPOINT_SHA256:
        raise ValueError("Les quatre essais doivent repartir du meme checkpoint V12 publie (epoque 50).")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    root = Path(__file__).resolve().parent
    for name in ("hausa_ocr/models.py", "hausa_ocr/augmentations.py"):
        normalized = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if normalized != config["source_sha256"][name]:
            raise ValueError(f"La recette V12 archivee a change : {name}.")
    if (config["version"] != "v12" or checkpoint["epoch"] != 50 or checkpoint["vocab"] != vocab or
            config["split_sha256"] != report["split_sha256"] or config["vocab_sha256"] != report["vocab_sha256"] or
            config["counts"] != report["counts"] or config["channels"] != 2 or config["cnn_convolutions"] != 9 or config["cnn_dropout"] != 0.05):
        raise ValueError("Poids V12, architecture, vocabulaire ou split incompatibles.")
    return checkpoint


def original_loader(dataset, rows, vocab, batch_size, device):
    clean = TrialDataset(dataset, rows, vocab)
    return DataLoader(clean, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda")


def snapshot(output):
    root = Path(__file__).resolve().parent
    destination = output / "source"
    hashes = {}
    for name in SOURCE_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        hashes[name] = sha256(target)
    return hashes


def assert_source(config):
    root = Path(__file__).resolve().parent
    for name, expected in config["source_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Source modifiee depuis le run : {name}. Utiliser les sources archivees de ce run.")


def eligible(validation, initial_cer):
    return validation["cer"] <= initial_cer + 1e-12


def better_selected(validation, incumbent, initial_cer):
    return eligible(validation, initial_cer) and (validation["wer"], validation["cer"]) < (incumbent["wer"], incumbent["cer"])


def make_checkpoint(model, optimizer, scheduler, config, vocab, epoch, validation, trackers, elapsed, peak, global_step):
    return dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                config=config, vocab=vocab, epoch=epoch, validation=validation, trackers=trackers, elapsed_seconds=elapsed,
                peak_vram_mib=peak, global_step=global_step, python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])


def epoch_reports(output, config, records, initial, trackers):
    units = dict(train_loss="example-weighted mean of effective-group objectives CTC + sampled MWER contribution",
                 train_ctc_loss="example-weighted mean of augmented microbatch CTC losses",
                 train_mwer_risk="mean exact CTC posterior-weighted word edit risk over sampled TRAIN examples only; zero if none",
                 train_sam_perturbed_loss="example-weighted group objective on perturbed parameters, or unperturbed loss for A/B",
                 mwer_examples="number of TRAIN views with the auxiliary MWER term, not the size of the training set",
                 val_cer="strict corpus character error ratio", val_wer="strict corpus word error ratio", val_uwer=UWER_DEFINITION,
                 val_exact_rate="ratio of exactly transcribed validation segments", seconds="training + validation before checkpoint saving",
                 epoch_peak_vram_mib="peak memory allocated by PyTorch this epoch, not total GPU memory",
                 lr_used="learning rate used", lr="learning rate for next epoch", optimizer_steps="updates in this epoch after accumulation",
                 val_cer_slope="CER ratio per epoch over at most five recent epochs", val_cer_window_gain="first minus last CER ratio in that window")
    write_json(output / "epochs.json", dict(version="v14", trial=config["trial"], selection=SELECTION, units=units,
               initial_validation=initial, completed_epochs=len(records), selected_epoch=trackers["selected_epoch"],
               source_history_sha256=sha256(output / "history.csv") if records else None, epochs=records))
    text = f"# V14-{config['trial']} — IX-World — historique complet\n\n"
    text += "Le CSV et le JSON conservent les mesures non arrondies. Les taux ci-dessous sont des pourcentages. Le train utilise 39 776 vues ; les taux CER/WER/UWER par époque sont ceux de la validation originale. Aucun test n'est évalué pendant l'apprentissage.\n\n"
    text += f"V12 initial : CER {initial['cer']:.4%}, WER {initial['wer']:.4%}. Époque retenue : {trackers['selected_epoch']} ; 0 signifie que les poids V12 initiaux ont été conservés faute de gain admissible.\n\n"
    text += "| Époque | Loss totale | CTC | Risque MWER | Vues MWER | VAL CER % | VAL WER % | VAL UWER % | LR | Temps s | Pic MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in records:
        text += f"| {r['epoch']} | {float(r['train_loss']):.6f} | {float(r['train_ctc_loss']):.6f} | {float(r['train_mwer_risk']):.6f} | {r['mwer_examples']} | {100*float(r['val_cer']):.4f} | {100*float(r['val_wer']):.4f} | {100*float(r['val_uwer']):.4f} | {float(r['lr_used']):g} | {float(r['seconds']):.1f} | {float(r['epoch_peak_vram_mib']):.1f} |\n"
    text += "\nLa loss totale comprend MWER seulement aux mises à jour échantillonnées de D. Le risque moyen est calculé sur ces vues, donc la loss totale de l'époque n'est pas simplement CTC + λ × la colonne risque. SAM réutilise les mêmes images, candidats et masques dropout pour sa seconde passe ; BatchNorm ne met ses statistiques à jour qu'à la première.\n"
    (output / "EPOCHS.md").write_text(text, encoding="utf-8", newline="\n")


def run_card(output, config, summary):
    text = f"# V14-{config['trial']} Hausa Ajami — IX-World\n\n"
    text += f"Affinage indépendant depuis V12 époque 50, SHA-256 `{config['initial_checkpoint_sha256']}`. Optimiseur neuf ; aucun poids des essais A/B/C ne sert à initialiser l'essai suivant.\n\n"
    text += f"Politique : `{config['augmentation']}` ; SAM={config['sam']}, MWER={config['mwer']}. Architecture V12 inchangée : neuf convolutions, BatchNorm, deux canaux gris/encre, BiLSTM deux couches, CTC. {config['parameter_count']} paramètres.\n\n"
    text += "Split figé : 4 972 train / 200 validation / 401 test, sources et groupes de copistes connus séparés ; sources 1/2 regroupées. Les quasi-doublons et copistes inconnus restent une limite.\n\n"
    text += f"Dataset `{config['dataset_revision']}` ; split `{config['split_sha256']}` ; vocabulaire `{config['vocab_sha256']}`.\n\n"
    text += f"Microbatch ≤ {config['batch_size']}, accumulation {config['accumulate']}, budget {config['max_batch_elements']} éléments N×2×96×W ; validation à batch {config['eval_batch_size']} comme le V12 initial. Float32, workers 0, LR {config['learning_rate']}, AdamW weight decay {config['weight_decay']}. Graine {config['seed']}, CUDA non strictement déterministe.\n\n"
    text += f"Sélection : minimum WER validation avec CER ≤ CER initial V12 ({config['initial_validation']['cer']:.4%}), puis CER en cas d'égalité. Époque retenue : {summary['selected_epoch']}. Époque 0 = V12 conservé. Les meilleurs CER et WER sans contrainte sont sauvegardés séparément et ne remplacent pas cette sélection.\n\n"
    text += f"Époques terminées : {summary['completed_epochs']} ; durée train + validation enregistrée {summary['elapsed_training_seconds']/3600:.2f} h, hors sauvegardes et bilan final ; pic {summary['peak_vram_mib']:.1f} MiB alloués par PyTorch, pas la mémoire totale occupée sur le GPU.\n\n"
    if config["sam"]:
        text += f"SAM L2, rayon {config['sam_rho']}, deux passes successives par mise à jour ; accumulation prise en compte avant perturbation, restauration exacte avant AdamW, une seule mise à jour des statistiques BatchNorm.\n\n"
    if config["mwer"]:
        text += f"MWER auxiliaire λ={config['mwer_weight']}, une application toutes les {config['mwer_every']} mises à jour, jusqu'à {config['mwer_samples']} vues TRAIN choisies uniformément dans le groupe accumulé. Beam {config['mwer_beam_width']}, n-best {config['mwer_nbest']}, top-k labels {config['mwer_token_top_k']} pour économiser le CPU ; recherche approchée, puis probabilités CTC recalculées exactement et différentiablement. Température {config['mwer_temperature']}. Le glouton et la référence TRAIN sont ajoutés aux candidats dédupliqués. Aucune référence de validation/test dans cette perte.\n\n"
    text += "Les huit familles d'augmentation restent celles de V12 ; B/C/D renouvellent leur graine à chaque époque. Le canal d'encre est recalculé après augmentation (fond gaussien rayon 5, gain 1,6). Les transformations peuvent altérer des points fins : leur innocuité linguistique n'est pas démontrée. Le décodage d'évaluation reste glouton CTC, sans lexique ni correction.\n\n"
    text += "| Partition | Images | CER | WER | UWER | Exact |\n|---|---:|---:|---:|---:|---:|\n"
    for part, values in summary["partitions"].items():
        text += f"| {part} | {values['examples']} | {values['cer']:.4%} | {values['wer']:.4%} | {values['uwer']:.4%} | {values['exact_rate']:.4%} |\n"
    text += f"\nUWER : {UWER_DEFINITION}\n\n"
    text += "Les trois taux sont des erreurs globales cumulées, pas des taux de réussite. Le train est réévalué une fois par image originale. Les 401 exemples du test sont évalués uniquement sur demande explicite --evaluate-test, après le choix de l'essai ; sinon aucun résultat de test n'est annoncé.\n\n"
    text += "Le code exact, ses empreintes, le protocole, les métriques, prédictions, historique CSV/JSON/Markdown et checkpoints sont conservés dans ce dossier. Aucun gain n'est garanti ; une seule graine ne suffit pas à une conclusion générale. Une augmentation du score train ne prouve pas une meilleure généralisation.\n\n"
    text += "Attribution code et entraînement : IX-World. Corpus et manuscrits : crédits et conditions du dépôt IntelligenceResearchLab/Hausa ; aucune nouvelle attribution ou licence n'est créée.\n"
    (output / "README.md").write_text(text, encoding="utf-8", newline="\n")


def final_scores(model, config, splits, vocab, report, args, output, device, include_test=False):
    checkpoint = torch.load(output / "selected.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_hash = sha256(output / "selected.pt")
    summary = dict(version="v14", trial=args.trial, selected_epoch=checkpoint["epoch"], selection=SELECTION,
                   checkpoint_sha256=checkpoint_hash, split_sha256=report["split_sha256"], dataset_revision=report["dataset_revision"],
                   test_evaluated=include_test, evaluated_at_utc=datetime.now(timezone.utc).isoformat(), uwer_definition=UWER_DEFINITION, partitions={})
    with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    summary.update(completed_epochs=len(history), elapsed_training_seconds=sum(float(r["seconds"]) for r in history),
                   peak_vram_mib=max((float(r["peak_vram_mib"]) for r in history), default=0.0))
    for part in (("train", "validation", "test") if include_test else ("train", "validation")):
        print(f"Evaluation finale {part.upper()} : {len(splits[part])} originaux, sans augmentation...", flush=True)
        metrics, predictions = evaluate(model, original_loader(args.dataset, splits[part], vocab, args.eval_batch_size, device), vocab, device)
        metrics.update(version="v14", trial=args.trial, evaluated_partition=part, augmented=False, selected_epoch=checkpoint["epoch"],
                       checkpoint_sha256=checkpoint_hash, split_sha256=report["split_sha256"], dataset_revision=report["dataset_revision"],
                       selection=SELECTION, uwer_definition=UWER_DEFINITION)
        for prediction in predictions:
            prediction["evaluated_partition"] = part
        summary["partitions"][part] = metrics
        write_json(output / f"{part}_metrics.json", metrics)
        with (output / f"{part}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=predictions[0])
            writer.writeheader()
            writer.writerows(predictions)
    write_json(output / "metrics.json", summary)
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["partition", "examples", "cer", "wer", "uwer", "exact_rate", "character_errors", "reference_characters", "word_errors", "reference_words", "normalized_word_errors", "normalized_reference_words", "selected_epoch", "checkpoint_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(values, partition=part) for part, values in summary["partitions"].items())
    run_card(output, config, summary)
    print(f"\nBILAN V14-{args.trial} | epoque retenue {checkpoint['epoch']} | test={include_test}")
    for part, values in summary["partitions"].items():
        print(f"{part:12s} {values['examples']:5d} | CER {values['cer']:.4%} | WER {values['wer']:.4%} | UWER {values['uwer']:.4%} | exact {values['exact_rate']:.4%}")
    print(f"Resultats : {output / 'metrics.csv'}", flush=True)
    return summary


def compare_trials(base, output=None):
    results = []
    common = None
    for trial in TRIALS:
        folder = base / f"v14_{trial}_seed42"
        if not (folder / "metrics.json").is_file():
            print(f"{trial}: pas encore termine.")
            continue
        cfg, summary = read(folder / "config.json"), read(folder / "metrics.json")
        require_protocol(dict(cfg, all_usable_images_used=True, separation_unit="source_and_known_scribe"))
        if cfg["initial_checkpoint_sha256"] != V12_CHECKPOINT_SHA256 or cfg["trial"] != trial or cfg["seed"] != 42:
            raise ValueError("Essais d'origine differente.")
        current = {key: cfg[key] for key in DEFAULTS if key != "epochs"}
        current.update(max_epochs=cfg["max_epochs"], source_sha256=cfg["source_sha256"], initial_validation=cfg["initial_validation"])
        if common is not None and current != common:
            raise ValueError("Budgets, hyperparametres, code ou validation initiale differents entre essais : comparaison non controlee.")
        common = current
        score = read(folder / "validation_metrics.json")
        if score != summary["partitions"]["validation"] or sha256(folder / "selected.pt") != score["checkpoint_sha256"]:
            raise ValueError("Checkpoint et validation desynchronises.")
        results.append(dict(trial=trial, folder=str(folder), **{f: score[f] for f in ("cer", "wer", "uwer", "exact_rate", "selected_epoch", "checkpoint_sha256")}))
    if not results:
        raise FileNotFoundError("Aucun essai V14 termine.")
    results.sort(key=lambda r: (r["wer"], r["cer"], r["trial"]))
    for r in results:
        print(f"{r['trial']} | CER {r['cer']:.4%} | WER {r['wer']:.4%} | UWER {r['uwer']:.4%} | epoch {r['selected_epoch']}")
    print(f"Meilleur compromis observe sur validation : {results[0]['trial']}. A confirmer apres les quatre essais et, idealement, plusieurs graines.")
    if output:
        write_json(output, dict(selection="validation_only_WER_then_CER", test_used=False, results=results))
    return results


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="V14 IX-World : essais A-D depuis les memes poids V12, GTX 1060 6 Go.")
    parser.add_argument("--trial", choices=TRIALS)
    parser.add_argument("--dataset", type=Path, default=base / "Hausa_repo_2026-10-15")
    parser.add_argument("--protocol", type=Path, default=base / "protocol_hausa_all_grouped_20261016")
    parser.add_argument("--initial-v12", type=Path, default=base / "runs_hausa_all_grouped_20261016" / "v12_seed42" / "best_cer.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=Path, default=base / "runs_hausa_v14_20261018")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Apres selection de l'essai : bilan train/validation/test du selected.pt, sans apprendre.")
    parser.add_argument("--check-only", action="store_true", help="Verifier V12, split et un batch de validation, sans apprendre ni creer un run.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--show-examples", type=int, default=5)
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=None)
    args = parser.parse_args()
    if args.compare:
        compare_trials(args.runs, args.compare_output)
        return
    if not args.trial:
        parser.error("Choisir --trial A, B, C ou D.")
    if sum((args.resume, args.evaluate_test, args.check_only)) > 1:
        parser.error("Choisir une seule action resume, evaluate-test ou check-only.")
    seed = args.seed if args.seed is not None else 42
    output = (args.output or args.runs / f"v14_{args.trial}_seed{seed}").resolve()
    previous = read(output / "config.json") if args.resume or args.evaluate_test else None
    # Resume/evaluation automatically execute the exact archived code after a Git update.
    archived = output / "source" / "train_v14.py"
    if previous and archived.is_file() and Path(__file__).resolve() != archived.resolve():
        forwarded = [sys.executable, "-X", "utf8", "-u", str(archived), "--dataset", previous["dataset"], "--protocol", previous["protocol_path"],
                     "--initial-v12", previous["initial_checkpoint_path"], "--runs", str(args.runs), "--output", str(output)] + sys.argv[1:]
        raise SystemExit(subprocess.call(forwarded))
    for name, default in DEFAULTS.items():
        value = getattr(args, name)
        saved_name = "max_epochs" if name == "epochs" else name
        if previous and value is not None and name != "epochs" and value != previous[saved_name]:
            parser.error(f"Reprise : --{name.replace('_', '-')} doit conserver la valeur initiale.")
        setattr(args, name, value if value is not None else previous[saved_name] if previous else default)
    for name in ("epochs", "min_epochs", "patience", "batch_size", "accumulate", "max_batch_elements", "eval_batch_size", "threads", "log_every", "mwer_every", "mwer_samples", "mwer_beam_width", "mwer_nbest"):
        if getattr(args, name) < 1:
            parser.error(f"Parametre positif requis : {name}")
    if (args.min_epochs > args.epochs or args.show_examples < 0 or args.mwer_nbest > args.mwer_beam_width or args.mwer_token_top_k < 0 or
            any(not math.isfinite(getattr(args, n)) or getattr(args, n) <= 0 for n in ("learning_rate", "sam_rho", "mwer_weight", "mwer_temperature")) or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Hyperparametres invalides.")
    if previous and previous["trial"] != args.trial:
        parser.error("Ne pas melanger les essais dans un meme dossier.")
    if not previous and not args.check_only and output.exists():
        raise FileExistsError(f"Le dossier existe deja : {output}. Utiliser --resume pour une interruption.")
    if args.resume and (output / "metrics.json").exists():
        raise ValueError("Cet essai est deja termine ; conserver ses resultats et utiliser un nouvel output pour une autre experience.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible ; verifier PyTorch ou ajouter --device cpu pour un controle.")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    args.dataset, args.protocol = args.dataset.resolve(), args.protocol.resolve()
    print("Verification des empreintes du split et du corpus...", flush=True)
    splits, vocab, report = load_protocol(args.dataset, args.protocol)
    require_protocol(report)
    model = build_model("v12", len(vocab))
    if previous:
        if previous["initial_checkpoint_sha256"] != V12_CHECKPOINT_SHA256 or sum(p.numel() for p in model.parameters()) != previous["parameter_count"]:
            raise ValueError("Origine ou architecture du run V14 incompatible.")
    else:
        initial_checkpoint = load_v12(args.initial_v12, vocab, report)
        model.load_state_dict(initial_checkpoint["model_state_dict"], strict=True)
        if sum(p.numel() for p in model.parameters()) != initial_checkpoint["config"]["parameter_count"]:
            raise ValueError("Architecture V12 modifiee.")
        del initial_checkpoint
    model.to(device)
    if args.check_only:
        batches = original_loader(args.dataset, splits["validation"], vocab, args.eval_batch_size, device)
        batch = next(iter(batches))
        metrics, predictions = evaluate(model, [batch], vocab, device)
        archive = args.initial_v12.parent / "validation_predictions.csv"
        if args.eval_batch_size == 20 and archive.is_file():
            with archive.open(encoding="utf-8-sig", newline="") as handle:
                expected = {(r["partition"], r["file_name"]): r["prediction"] for r in csv.DictReader(handle)}
            if any(expected.get((r["partition"], r["file_name"])) != r["prediction"] for r in predictions):
                raise ValueError("Les predictions initiales ne reproduisent pas celles de V12 archive.")
            print(f"Les {len(predictions)} predictions reproduisent exactement le premier batch V12 archive.", flush=True)
        print(f"V14-{args.trial} verifie : poids V12 stricts, meme split, {metrics['examples']} predictions de validation, aucun apprentissage.", flush=True)
        return
    validation_loader = original_loader(args.dataset, splits["validation"], vocab, args.eval_batch_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    records, start_epoch, elapsed, peak, global_step = [], 1, 0.0, 0.0, 0
    if previous:
        config = previous
        assert_source(config)
        if sha256(output / "split.json") != report["split_sha256"] or sha256(output / "vocab.json") != report["vocab_sha256"] or sha256(output / "protocol.json") != sha256(args.protocol / "protocol.json"):
            raise ValueError("Protocole archive different du protocole courant.")
        checkpoint = torch.load(output / ("last.pt" if args.resume else "selected.pt"), map_location="cpu", weights_only=False)
        if checkpoint["config"] != config or checkpoint["vocab"] != vocab:
            raise ValueError("Checkpoint et configuration V14 differents.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        initial, trackers = config["initial_validation"], checkpoint["trackers"]
        if args.evaluate_test:
            final_scores(model, config, splits, vocab, report, args, output, device, include_test=True)
            return
        elapsed, peak, global_step = checkpoint["elapsed_seconds"], checkpoint["peak_vram_mib"], checkpoint["global_step"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch = checkpoint["epoch"] + 1
        with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
            raw_records = list(csv.DictReader(handle))
        for row in raw_records:
            records.append({key: None if value == "" else int(value) if key in ("epoch", "optimizer_steps", "mwer_examples", "selected_epoch", "eligible") else float(value) for key, value in row.items()})
        if not records or records[-1]["epoch"] != checkpoint["epoch"] or args.epochs < checkpoint["epoch"]:
            raise ValueError("Historique et checkpoint de reprise incoherents.")
        if args.epochs != config["max_epochs"]:
            raise ValueError("Conserver le budget max-epochs initial pour une reprise comparable des essais.")
    else:
        print("Validation initiale des poids V12 ; le test reste ferme...", flush=True)
        initial, _ = evaluate(model, validation_loader, vocab, device)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        config = dict(author=AUTHOR, source_repository=SOURCE_REPOSITORY, source_commit=commit, version="v14", trial=args.trial,
                      dataset=str(args.dataset), protocol_path=str(args.protocol), dataset_repo=report["dataset_repo"], dataset_revision=report["dataset_revision"],
                      protocol_id=report["protocol_id"], split_sha256=report["split_sha256"], vocab_sha256=report["vocab_sha256"], counts=report["counts"],
                      initial_checkpoint_path=str(args.initial_v12.resolve()), initial_checkpoint_sha256=sha256(args.initial_v12), initial_checkpoint_epoch=50,
                      from_scratch=False, optimizer_reinitialized=True, initial_validation=initial, selection=SELECTION,
                      **TRIALS[args.trial], max_epochs=args.epochs, **{k: getattr(args, k) for k in DEFAULTS if k != "epochs"},
                      height=96, channels=2, cnn_convolutions=9, cnn_dropout=0.05, lstm_layers=2, lstm_hidden_per_direction=256,
                      packed_lstm=False, normalization="BatchNorm2d", rtl_mirror=True, ink_blur_radius=5.0, ink_gain=1.6,
                      views_per_image=8, virtual_train_size=len(splits["train"]) * 8, parameter_count=sum(p.numel() for p in model.parameters()),
                      precision="float32", optimizer="AdamW", gradient_clip=5.0, evaluation_decoding="greedy_CTC_no_language_model",
                      test_policy="explicit_evaluate_test_only_after_trial_selection", uwer_definition=UWER_DEFINITION,
                      python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
        output.mkdir(parents=True)
        config["source_sha256"] = snapshot(output)
        write_json(output / "config.json", config)
        write_json(output / "initial_validation.json", initial)
        for name in ("split.json", "vocab.json", "protocol.json", "quarantine.json", "source_groups.json", "excluded_public_train.json"):
            shutil.copy2(args.protocol / name, output / name)
        trackers = dict(selected_validation=initial, selected_epoch=0, best_cer=initial["cer"], best_wer=initial["wer"],
                        patience_best=initial["wer"], without_gain=0)
        saved = make_checkpoint(model, optimizer, scheduler, config, vocab, 0, initial, trackers, elapsed, peak, global_step)
        for name in ("selected.pt", "best_cer.pt", "best_wer.pt"):
            save_checkpoint(output / name, saved)
    train_data = TrialDataset(args.dataset, splits["train"], vocab, config["augmentation"], args.seed)
    sampler = TensorBudgetSampler(train_data, args.batch_size, args.max_batch_elements, args.seed)
    batches = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_data.epoch = sampler.epoch = epoch
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            beginning, lr = time.monotonic(), optimizer.param_groups[0]["lr"]
            print(f"\nV14-{args.trial} EPOCH {epoch}/{args.epochs} | {len(batches)} microbatches | SAM={config['sam']} MWER={config['mwer']}", flush=True)
            stats, global_step = train_epoch(model, batches, optimizer, vocab, args, device, epoch, global_step)
            val, predictions = evaluate(model, validation_loader, vocab, device)
            scheduler.step(val["cer"])
            seconds = time.monotonic() - beginning
            elapsed += seconds
            epoch_peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
            peak = max(peak, epoch_peak)
            improve_cer, improve_wer = val["cer"] < trackers["best_cer"], val["wer"] < trackers["best_wer"]
            improved = better_selected(val, trackers["selected_validation"], initial["cer"])
            trackers["best_cer"], trackers["best_wer"] = min(trackers["best_cer"], val["cer"]), min(trackers["best_wer"], val["wer"])
            if improved:
                trackers.update(selected_validation=val, selected_epoch=epoch)
            if eligible(val, initial["cer"]) and val["wer"] < trackers["patience_best"] - 0.0005:
                trackers.update(patience_best=val["wer"], without_gain=0)
            else:
                trackers["without_gain"] += 1
            record = dict(epoch=epoch, train_loss=stats["total"] / stats["examples"], train_ctc_loss=stats["ctc_sum"] / stats["examples"],
                          train_mwer_risk=stats["risk_sum"] / max(1, stats["risk_examples"]), mwer_examples=stats["risk_examples"],
                          train_sam_perturbed_loss=stats["sam_loss"] / stats["examples"], optimizer_steps=stats["updates"],
                          max_gradient_norm=stats["max_gradient_norm"], val_cer=val["cer"], val_wer=val["wer"], val_uwer=val["uwer"],
                          val_exact_rate=val["exact_rate"], lr_used=lr, lr=optimizer.param_groups[0]["lr"], seconds=seconds,
                          vram_allocated_mib=torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0.0,
                          epoch_peak_vram_mib=epoch_peak, peak_vram_mib=peak, eligible=int(eligible(val, initial["cer"])), selected_epoch=trackers["selected_epoch"])
            slope, gain = cer_trend(records + [record])
            record.update(val_cer_slope=slope, val_cer_window_gain=gain)
            append_history(output / "history.csv", records, record)
            saved = make_checkpoint(model, optimizer, scheduler, config, vocab, epoch, val, trackers, elapsed, peak, global_step)
            for keep, name in ((improved, "selected.pt"), (improve_cer, "best_cer.pt"), (improve_wer, "best_wer.pt"), (True, "last.pt")):
                if keep:
                    save_checkpoint(output / name, saved)
            epoch_reports(output, config, records, initial, trackers)
            print("\n" + "=" * 70)
            print(f"V14-{args.trial} | EPOCH {epoch:03d}/{args.epochs} | CTC {record['train_ctc_loss']:.6f} | MWER {record['train_mwer_risk']:.6f} N={record['mwer_examples']}")
            print(f"VAL CER {val['cer']:.4%} | WER {val['wer']:.4%} | UWER {val['uwer']:.4%} | exact {val['exact_rate']:.4%}")
            print(f"LR {lr:g} -> {record['lr']:g} | temps {seconds:.1f}s | pic {epoch_peak:.1f} MiB | pente CER {slope} | gain {gain}")
            print(f"Train reel {len(splits['train'])} | vues {len(train_data)} | canaux 2 | CNN 9 | best CER {trackers['best_cer']:.4%} | best WER {trackers['best_wer']:.4%}")
            print(f"Compromis retenu @{trackers['selected_epoch']} | sans gain WER admissible {trackers['without_gain']}/{args.patience}", flush=True)
            for row in predictions[:args.show_examples]:
                print(f"FILE: {row['file_name']}\nREF : {row['reference']}\nPRED: {row['prediction']}", flush=True)
            if epoch >= args.min_epochs and trackers["without_gain"] >= args.patience:
                print("Arret anticipe sur le WER de validation sous contrainte CER.", flush=True)
                break
    except KeyboardInterrupt:
        print("Interruption : checkpoints des epoques completes conserves. Reprendre le meme essai avec --resume.", flush=True)
        raise SystemExit(130) from None
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("VRAM insuffisante. Liberer la carte, ou utiliser un NOUVEAU dossier avec --batch-size 2 --accumulate 10 --eval-batch-size 10. Appliquer ces memes reglages aux quatre essais pour les comparer.") from None
    final_scores(model, config, splits, vocab, report, args, output, device)


if __name__ == "__main__":
    main()
