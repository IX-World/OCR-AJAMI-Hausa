"""IX-World: compare V10 greedy and CTC beam decoding on validation only."""
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hausa_ocr.ctc_beam import prefix_beam_search, tokens_to_text
from hausa_ocr.data import LineDataset, collate, decode, edit_distance
from hausa_ocr.metrics import UWER_DEFINITION, normalize_uwer
from hausa_ocr.models import build_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_validation(run, dataset=None):
    config, protocol, vocab = (read_json(run / name) for name in ("config.json", "protocol.json", "vocab.json"))
    if config["version"] != "v10" or config["channels"] != 1:
        raise ValueError("Ce script est reserve au checkpoint V10 gris.")
    for name, key in (("split.json", "split_sha256"), ("vocab.json", "vocab_sha256")):
        if digest(run / name) != config[key] or config[key] != protocol[key]:
            raise ValueError(f"Manifeste modifie : {name}")
    if config.get("protocol_sha256") and digest(run / "protocol.json") != config["protocol_sha256"]:
        raise ValueError("Protocole modifie depuis l'entrainement.")
    if protocol.get("separation_unit") != "source_and_known_scribe" or not protocol.get("all_usable_images_used"):
        raise ValueError("Le protocole complet separe par provenance est requis.")
    # The frozen split contains all roles, but only validation is extracted.
    # No test image, prediction, metric, or reference is used for decoding.
    rows = read_json(run / "split.json")["validation"]
    if len(rows) != config["counts"]["validation"] or not rows:
        raise ValueError("Effectif de validation incorrect.")
    if len({(r["partition"], r["file_name"]) for r in rows}) != len(rows):
        raise ValueError("Identifiants de validation dupliques.")
    root = Path(dataset or config["dataset"]).resolve()
    for row in rows:
        if row.get("split") != "validation" or protocol["source_to_group"].get(str(row["source"])) != row.get("group_id"):
            raise ValueError("Une ligne n'appartient pas a la validation archivee.")
        path = (root / "data" / row["partition"] / row["file_name"]).resolve()
        if not path.is_relative_to(root / "data") or digest(path) != row["file_sha256"]:
            raise ValueError(f"Image de validation absente ou modifiee : {row['file_name']}")
    baseline = read_json(run / "validation_metrics.json")
    if baseline.get("evaluated_partition") != "validation" or baseline.get("augmented") or baseline["examples"] != len(rows):
        raise ValueError("Bilan historique de validation incorrect.")
    if baseline["split_sha256"] != config["split_sha256"]:
        raise ValueError("Le bilan de validation utilise un autre split.")
    with (run / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        published = list(csv.DictReader(handle))
    expected = {(r["partition"], r["file_name"]): r["transcript"] for r in rows}
    if len(published) != len(rows) or {(r["partition"], r["file_name"]): r["reference"] for r in published} != expected:
        raise ValueError("Les predictions de validation ne couvrent pas le manifeste.")
    return config, rows, vocab, root, baseline, published


def score(predictions, rows):
    if len(predictions) != len(rows):
        raise ValueError("Nombre de predictions incorrect.")
    result = dict(examples=len(rows), character_errors=0, reference_characters=0, word_errors=0,
                  reference_words=0, normalized_word_errors=0, normalized_reference_words=0, exact=0)
    for prediction, row in zip(predictions, rows):
        reference = row["transcript"]
        result["character_errors"] += edit_distance(reference, prediction)
        result["reference_characters"] += len(reference)
        result["word_errors"] += edit_distance(reference.split(), prediction.split())
        result["reference_words"] += len(reference.split())
        normalized = normalize_uwer(reference).split()
        result["normalized_word_errors"] += edit_distance(normalized, normalize_uwer(prediction).split())
        result["normalized_reference_words"] += len(normalized)
        result["exact"] += reference == prediction
    for metric, numerator, denominator in (("cer", "character_errors", "reference_characters"), ("wer", "word_errors", "reference_words"), ("uwer", "normalized_word_errors", "normalized_reference_words"), ("exact_rate", "exact", "examples")):
        result[metric] = result[numerator] / max(1, result[denominator])
    return result


def validation_emissions(run, root, rows, vocab, config, baseline, device, output, threads):
    checkpoint_path = run / "best_cer.pt"
    checkpoint_sha = digest(checkpoint_path)
    if checkpoint_sha != baseline["checkpoint_sha256"]:
        raise ValueError("Les poids ne correspondent pas au bilan de validation.")
    sources = {name: digest(Path(__file__).resolve().parent / name) for name in ("hausa_ocr/models.py", "hausa_ocr/data.py")}
    for name in sources:
        archived = run / "source" / name
        current = Path(__file__).resolve().parent / name
        if digest(archived) != config["source_sha256"][name] or ast.dump(ast.parse(current.read_text(encoding="utf-8"))) != ast.dump(ast.parse(archived.read_text(encoding="utf-8"))):
            raise ValueError(f"Implementation differente du modele evalue : {name}")
    identity = dict(checkpoint_sha256=checkpoint_sha, split_sha256=config["split_sha256"],
                    validation_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                    source_sha256=sources, inference_script_sha256=digest(__file__), partition="validation", examples=len(rows),
                    batch_size=config["batch_size"], device=device, torch=str(torch.__version__), threads=threads)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = output / f"validation_emissions_{identity_sha[:20]}.pt"
    ids = [f"{r['partition']}/{r['file_name']}" for r in rows]
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["ids"] != ids:
            raise ValueError("Cache d'emissions incompatible.")
        emissions = saved["emissions"]
        print(f"Cache de validation reutilise : {cache.name}", flush=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != baseline["selected_epoch"] or checkpoint["vocab"] != vocab:
            raise ValueError("Checkpoint ou vocabulaire incompatible.")
        for field in ("version", "split_sha256", "vocab_sha256", "channels", "classes", "batch_size"):
            if checkpoint["config"][field] != config[field]:
                raise ValueError(f"Configuration du checkpoint differente : {field}")
        model = build_model("v10", len(vocab)).to(device).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        data = LineDataset(root, rows, vocab, "v10", training=False)
        batches = DataLoader(data, batch_size=config["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)
        emissions = []
        with torch.inference_mode():
            for number, batch in enumerate(batches, 1):
                probabilities = model(batch["images"].to(device)).log_softmax(-1).cpu()
                for i, width in enumerate(batch["widths"]):
                    emissions.append(probabilities[i, :int(width) // 4].clone())
                print(f"Inference VALIDATION : {len(emissions)}/{len(rows)} images | batch {number}/{len(batches)}", flush=True)
        temporary = cache.with_suffix(f".{os.getpid()}.tmp")
        torch.save(dict(identity=identity, ids=ids, emissions=emissions), temporary)
        temporary.replace(cache)
        del model
    if len(emissions) != len(rows):
        raise ValueError("Cache de validation incomplet.")
    for matrix, row in zip(emissions, rows):
        width = max(4, round(row["width"] * 96 / row["height"]))
        if tuple(matrix.shape) != (width // 4, len(vocab)) or not torch.isfinite(matrix).all():
            raise ValueError("Dimensions ou valeurs du cache incorrectes.")
    return emissions, dict(identity, cache_path=str(cache), cache_sha256=digest(cache))


def main(argv=None):
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="V10 : comparaison glouton / beam search, uniquement sur la validation ; aucun apprentissage.")
    parser.add_argument("--run", type=Path, default=base / "runs_hausa_all_grouped_20261016" / "v10_seed42")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=base / "validation_beam_v10")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[10, 25])
    parser.add_argument("--nbest", type=int, default=5)
    parser.add_argument("--token-top-k", type=int, default=0, help="0 = tous les caracteres ; >0 ajoute un elagage approximatif.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if any(width < 1 or width > 256 for width in args.beam_widths) or args.nbest < 1 or args.token_top_k < 0 or args.threads < 1:
        parser.error("Parametres de decodage invalides.")
    args.run, args.output = args.run.resolve(), args.output.resolve()
    config, rows, vocab, root, baseline, published = load_validation(args.run, args.dataset)
    if args.output.is_relative_to(args.run) or args.output.is_relative_to(root):
        parser.error("Les resultats doivent etre separes du run et du dataset.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponible ; utiliser --device cpu.")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"V10 | VALIDATION SEULE : {len(rows)} images | batch historique : {config['batch_size']} | {args.device.upper()}", flush=True)
    print("Poids figes, aucune augmentation, aucun apprentissage, aucun acces aux images/mesures du test experimental.", flush=True)
    emissions, provenance = validation_emissions(args.run, root, rows, vocab, config, baseline, args.device, args.output, args.threads)
    greedy_beginning = time.monotonic()
    greedy = [decode(matrix.argmax(-1), vocab) for matrix in emissions]
    greedy_seconds = time.monotonic() - greedy_beginning
    expected = {(r["partition"], r["file_name"]): r["prediction"] for r in published}
    differences = [r["file_name"] for r, p in zip(rows, greedy) if expected[(r["partition"], r["file_name"])] != p]
    if differences:
        raise ValueError(f"Le glouton ne reproduit pas les predictions historiques ({len(differences)} differences, ex. {differences[:3]}). Comparaison annulee ; verifier device/versions avant de conclure.")
    all_predictions = {"greedy": greedy}
    scores = {"greedy": dict(score(greedy, rows), method="greedy", decode_seconds=greedy_seconds)}
    for field in ("character_errors", "word_errors", "normalized_word_errors", "reference_characters", "reference_words", "exact", "examples"):
        if scores["greedy"][field] != baseline[field]:
            raise ValueError(f"Compteur glouton different du bilan publie : {field}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    experiment = args.output / stamp
    experiment.mkdir()
    with (experiment / "nbest.jsonl").open("w", encoding="utf-8", newline="\n") as candidates:
        for width in dict.fromkeys(args.beam_widths):
            method = f"beam_{width}"
            predictions = []
            beginning = time.monotonic()
            for index, (matrix, row) in enumerate(zip(emissions, rows), 1):
                hypotheses = prefix_beam_search(matrix.tolist(), width, vocab["<BLANK>"], min(args.nbest, width), args.token_top_k)
                texts = [{"prediction": tokens_to_text(tokens, vocab), "log_ctc_probability": value} for tokens, value in hypotheses]
                predictions.append(texts[0]["prediction"])
                candidates.write(json.dumps(dict(method=method, evaluated_partition="validation", sample_id=f"{row['partition']}/{row['file_name']}", hypotheses=texts), ensure_ascii=False) + "\n")
                if index == 1 or index % 20 == 0 or index == len(rows):
                    print(f"{method} : {index}/{len(rows)} | {time.monotonic()-beginning:.1f}s", flush=True)
            all_predictions[method] = predictions
            scores[method] = dict(score(predictions, rows), method=method, decode_seconds=time.monotonic()-beginning)
    for values in scores.values():
        values["delta_cer_points"] = 100 * (values["cer"] - scores["greedy"]["cer"])
        values["delta_wer_points"] = 100 * (values["wer"] - scores["greedy"]["wer"])
    if digest(args.run / "best_cer.pt") != provenance["checkpoint_sha256"]:
        raise ValueError("Le checkpoint a change pendant l'experience ; comparaison annulee.")
    source_files = ("eval_beam_v10_validation.py", "hausa_ocr/ctc_beam.py", "hausa_ocr/models.py", "hausa_ocr/data.py", "hausa_ocr/metrics.py")
    source_hashes = {name: digest(Path(__file__).resolve().parent / name) for name in source_files}
    for name in source_files:
        target = experiment / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parent / name, target)
    summary = dict(author="IX-World", version="v10", evaluated_partition="validation", test_used=False, training_performed=False,
                   language_model=None, lexicon=None, semantic_learning=False, checkpoint_epoch=baseline["selected_epoch"],
                   baseline_predictions_reproduced=True, token_top_k=args.token_top_k, nbest=args.nbest,
                   uwer_definition=UWER_DEFINITION, provenance=provenance, source_sha256=source_hashes, methods=scores)
    write_json(experiment / "summary.json", summary)
    with (experiment / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores["greedy"]))
        writer.writeheader()
        writer.writerows(scores.values())
    with (experiment / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "evaluated_partition", "public_origin", "file_name", "reference", "prediction"])
        writer.writeheader()
        for method, predictions in all_predictions.items():
            writer.writerows(dict(method=method, evaluated_partition="validation", public_origin=r["partition"], file_name=r["file_name"], reference=r["transcript"], prediction=p) for r, p in zip(rows, predictions))
    report = "# V10 : glouton et beam search sur la validation\n\nPoids figés, aucun réentraînement et aucune évaluation du test expérimental. Aucun modèle de langue ni apprentissage sémantique dans cet essai. Le glouton reproduit exactement les prédictions historiques de validation.\n\n"
    report += "| Méthode | CER | WER | UWER | Exact | Écart WER (points) |\n|---|---:|---:|---:|---:|---:|\n"
    print("\nBILAN : VALIDATION UNIQUEMENT")
    for method, values in scores.items():
        report += f"| {method} | {values['cer']:.4%} | {values['wer']:.4%} | {values['uwer']:.4%} | {values['exact_rate']:.4%} | {values['delta_wer_points']:+.4f} |\n"
        print(f"{method:12s} CER {values['cer']:.4%} | WER {values['wer']:.4%} | UWER {values['uwer']:.4%}")
    report += "\nLe beam search approxime les probabilités CTC par un ensemble limité de préfixes. Les paramètres sont étudiés sur la validation ; ces mesures ne sont pas un résultat de test indépendant. `nbest.jsonl` contient des hypothèses non choisies à partir des références, pour préparer un futur reclassement linguistique.\n"
    (experiment / "README.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"Resultats : {experiment}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruption : les poids restent inchanges ; le cache complet, s'il existe, pourra etre reutilise.", flush=True)
        raise SystemExit(130)
