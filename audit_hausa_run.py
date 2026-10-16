"""IX-World: audit an existing run without changing its data or weights."""
import argparse
import csv
import json
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

for location in (Path(__file__).resolve().parent, Path(__file__).resolve().parent / "evaluation_source"):
    if (location / "hausa_ocr" / "metrics.py").is_file():
        sys.path.insert(0, str(location))
        break

from hausa_ocr.data import edit_distance
from hausa_ocr.metrics import normalize_uwer
from prepare_hausa import read_rows, sha256, write_json
from hausa_ocr.protocol import validate_partitions


def sample_id(row):
    return row["partition"], row["file_name"]


def text_record(row):
    return row["file_name"], unicodedata.normalize("NFC", row["transcript"]), str(row["source"])


def add_scores(group, row):
    reference, prediction = row["reference"], row["prediction"]
    group["examples"] += 1
    group["character_errors"] += edit_distance(reference, prediction)
    group["reference_characters"] += len(reference)
    group["word_errors"] += edit_distance(reference.split(), prediction.split())
    group["reference_words"] += len(reference.split())
    group["normalized_word_errors"] += edit_distance(normalize_uwer(reference).split(), normalize_uwer(prediction).split())
    group["normalized_reference_words"] += len(normalize_uwer(reference).split())
    group["exact"] += int(reference == prediction)


def ratios(counts):
    return dict(counts, cer=counts["character_errors"] / max(1, counts["reference_characters"]), wer=counts["word_errors"] / max(1, counts["reference_words"]), uwer=counts["normalized_word_errors"] / max(1, counts["normalized_reference_words"]), exact_rate=counts["exact"] / max(1, counts["examples"]))


def audit(run, dataset, output):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    strict = validate_partitions(split, protocol)
    excluded = json.loads((run / "excluded_public_train.json").read_text(encoding="utf-8")) if strict else []
    if sha256(run / "split.json") != config["split_sha256"] or summary["split_sha256"] != config["split_sha256"]:
        raise ValueError("Split modifie depuis l'experience.")
    if sha256(run / "best_cer.pt") != summary["checkpoint_sha256"]:
        raise ValueError("Le checkpoint ne correspond plus aux predictions.")
    for part, field in (("train", "corrected_train_metadata_sha256"), ("test", "test_metadata_sha256")):
        if sha256(dataset / "data" / part / "metadata.jsonl") != protocol[field]:
            raise ValueError("Metadonnees publiques modifiees.")
        if protocol.get("all_usable_images_used",False):
            actual=[r for entries in split.values() for r in entries if r["partition"]==part]
        else:
            actual = split["test"] if part == "test" else split["train"] + split["validation"]
            if part == "train":actual = actual + excluded
        public = read_rows(dataset / "data" / part / "metadata.jsonl")
        if sorted(map(text_record, actual)) != sorted(map(text_record, public)):
            raise ValueError("Le split ne couvre pas les metadonnees publiques corrigees.")
        if any(row["partition"] != part for row in actual):
            raise ValueError("Origine train/test incorrecte.")
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{a}_{b}"] = {"identifiers": len(set(map(sample_id, split[a])) & set(map(sample_id, split[b]))), "pixel_hashes": len({r["pixel_sha256"] for r in split[a]} & {r["pixel_sha256"] for r in split[b]})}
    if any(any(row.values()) for row in overlaps.values()):
        raise ValueError("Chevauchement d'images entre partitions.")
    train_words = {word for row in split["train"] for word in row["transcript"].split()}
    train_normalized_words = {word for row in split["train"] for word in normalize_uwer(row["transcript"]).split()}
    train_texts = {row["transcript"] for row in split["train"]}
    train_sources = {str(row["source"]) for row in split["train"]}
    composition, analyses = {}, {}
    for part, rows in split.items():
        words = [word for row in rows for word in row["transcript"].split()]
        normalized = [word for row in rows for word in normalize_uwer(row["transcript"]).split()]
        composition[part] = {
            "examples": len(rows), "reference_words": len(words), "reference_characters": sum(len(r["transcript"]) for r in rows),
            "mean_words_per_segment": len(words) / len(rows), "median_words_per_segment": statistics.median(len(r["transcript"].split()) for r in rows),
            "single_word_segments": sum(len(r["transcript"].split()) == 1 for r in rows),
            "strict_oov_word_occurrences": sum(word not in train_words for word in words),
            "normalized_oov_word_occurrences": sum(word not in train_normalized_words for word in normalized),
            "reference_text_seen_in_train": sum(r["transcript"] in train_texts for r in rows),
            "source_counts": dict(Counter(str(r["source"]) for r in rows)),
            "segments_from_sources_absent_from_train": sum(str(r["source"]) not in train_sources for r in rows),
        }
        with (run / f"{part}_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        expected = {sample_id(row): row["transcript"] for row in rows}
        if len(predictions) != len(expected) or {sample_id(row): row["reference"] for row in predictions} != expected:
            raise ValueError(f"Couverture ou references incorrectes : {part}")
        groups = {name: defaultdict(Counter) for name in ("all", "by_source", "by_source_exposure", "by_word_count", "by_reference_exposure")}
        for row in predictions:
            if edit_distance(row["reference"], row["prediction"]) != int(row["character_errors"]) or edit_distance(row["reference"].split(), row["prediction"].split()) != int(row["word_errors"]):
                raise ValueError("Compteurs individuels incorrects.")
            n_words = len(row["reference"].split())
            categories = {"all": "all", "by_source": str(row["source"]), "by_source_exposure": "seen" if str(row["source"]) in train_sources else "unseen", "by_word_count": "1" if n_words == 1 else "2-4" if n_words <= 4 else "5+", "by_reference_exposure": "seen" if row["reference"] in train_texts else "unseen"}
            for category, key in categories.items():
                add_scores(groups[category][key], row)
        analyses[part] = {name: {key: ratios(value) for key, value in members.items()} for name, members in groups.items()}
        recomputed = analyses[part]["all"]["all"]
        for field, value in recomputed.items():
            if abs(value - summary["partitions"][part][field]) > 1e-12:
                raise ValueError(f"Score {field} non reproductible pour {part}.")
    result = {"author": "IX-World", "version": config["version"], "dataset_revision": config["dataset_revision"], "split_sha256": config["split_sha256"], "checkpoint_sha256": summary["checkpoint_sha256"], "audit_code_sha256": sha256(__file__), "counts": config["counts"], "public_test_preserved": True, "public_train_coverage_complete": True, "excluded_public_train_count":len(excluded), "group_disjoint":strict, "cross_partition_overlap": overlaps, "pixel_overlap_method": "stored grayscale pixel hashes from preparation; original file hashes checked by evaluator", "near_duplicates_checked": False, "composition": composition, "error_analysis": analyses}
    result["public_test_preserved"]=protocol.get("public_test_unchanged",True)
    (output / "split_audit.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    result["markdown_sha256"] = sha256(output / "split_audit.md")
    write_json(output / "split_audit.json", result)
    print(json.dumps({"composition": composition, "test_by_source_exposure": analyses["test"]["by_source_exposure"], "validation_by_word_count": analyses["validation"]["by_word_count"], "test_by_word_count": analyses["test"]["by_word_count"]}, ensure_ascii=False, indent=2))
    return result


def markdown(report):
    parts = report["composition"]
    val, test = parts["validation"], parts["test"]
    text = "## Audit du split et interpretation de l'ecart validation/test\n\n"
    text += f"Audit du checkpoint `{report['checkpoint_sha256']}` et du split `{report['split_sha256']}`.\n\n"
    text += "Les exemples et leurs references couvrent exactement le corpus corrige, en tenant compte des exclusions documentees. Aucun identifiant ni hash de pixels exact n'est partage entre partitions. La recherche de quasi-doublons n'a pas ete realisee. Les CER, WER et UWER ont ete recalcules depuis les predictions et leurs references : ils correspondent aux fichiers de scores.\n\n"
    text += "| Composition | Train | Validation | Test |\n|---|---:|---:|---:|\n"
    for label, field, form in (("Segments", "examples", "d"), ("Mots", "reference_words", "d"), ("Mots moyens par segment", "mean_words_per_segment", ".2f"), ("Segments d'un seul mot", "single_word_segments", "d"), ("Mots stricts absents du train (occurrences)", "strict_oov_word_occurrences", "d"), ("References completes deja presentes dans le train", "reference_text_seen_in_train", "d"), ("Segments de sources absentes du train", "segments_from_sources_absent_from_train", "d")):
        text += "| " + label + " | " + " | ".join(format(parts[p][field], form) for p in ("train", "validation", "test")) + " |\n"
    text += f"\nLa validation contient {val['single_word_segments']/val['examples']:.2%} de segments d'un seul mot, contre {test['single_word_segments']/test['examples']:.2%} au test. {test['segments_from_sources_absent_from_train']} exemples de test proviennent de sources absentes du train ; la repartition complete par source est dans split_audit.json. Ces differences de composition doivent etre prises en compte pour interpreter les scores de validation et du test du protocole.\n\n"
    text += "| Groupe | Validation N / mots | Validation WER | Test N / mots | Test WER |\n|---|---:|---:|---:|---:|\n"
    for category, key, label in (("by_source_exposure", "seen", "Source presente au train"), ("by_source_exposure", "unseen", "Source absente du train"), ("by_word_count", "1", "Segments d'un mot"), ("by_word_count", "2-4", "Segments de 2 a 4 mots"), ("by_word_count", "5+", "Segments de 5 mots et plus"), ("by_reference_exposure", "seen", "Reference complete vue au train"), ("by_reference_exposure", "unseen", "Reference complete non vue au train")):
        cells = []
        for part in ("validation", "test"):
            value = report["error_analysis"][part][category].get(key)
            cells.extend([f"{value['examples']} / {value['reference_words']}", f"{value['wer']:.2%}"] if value else ["0 / 0", "non mesure"])
        text += f"| {label} | " + " | ".join(cells) + " |\n"
    text += "\nCes groupes decrivent des associations, sans isoler une cause unique. La longueur ne majore pas mecaniquement le WER, qui est normalise par le nombre de mots ; elle change notamment les conditions visuelles de reconnaissance. La validation intervient aussi dans la selection du checkpoint et ne constitue pas un second test independant. Le WER des sources connues et inconnues du test doit etre examine avant d'attribuer l'ecart aux seules sources. Les conventions d'annotation et la qualite visuelle n'ont pas ete relues integralement dans cet audit.\n\n"
    text += "Le split et le checkpoint documentes restent figes pour comparer V9 a V12. Les decisions de repartition reposent sur les groupes de provenance et leurs effectifs. Le fichier de protocole indique si le test public initial est conserve ou reconstruit.\n"
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit(args.run, args.dataset, args.output)
