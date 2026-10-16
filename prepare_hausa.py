import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REVISION = "1fb85fa070b6cf6432f1d45d21f78851d57f8944"
PATH_FIXES = {
    "ajami_00462png": "ajami_00462.png",
    "ajami_01243png": "ajami_01243.png",
    "ajami_01O53.png": "ajami_01053.png",
}
QUARANTINE = {"a_ajami_03400.png", "a_ajami_03401.png"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def pixel_hash(path):
    with Image.open(path) as image:
        image = image.convert("L")
        return hashlib.sha256(str(image.size).encode("ascii") + image.tobytes()).hexdigest()


def corrected_rows(rows):
    clean, changes, excluded, seen = [], [], [], {}
    for number, original in enumerate(rows, 1):
        row = dict(original)
        name = row["file_name"]
        if name in PATH_FIXES:
            row["file_name"] = PATH_FIXES[name]
            changes.append({"line": number, "before": name, "after": row["file_name"], "reason": "filename_typo"})
        if name == "ajami_01299.png" and row["transcript"] == "ranar":
            if number < 2 or number >= len(rows) or rows[number - 2]["file_name"] != "ajami_01288.png" or rows[number]["file_name"] != "ajami_01290.png":
                raise ValueError("La reference 01289 exige une nouvelle verification : voisinage modifie.")
            row["file_name"] = "ajami_01289.png"
            changes.append({"line": number, "before": name, "after": row["file_name"], "reason": "sequence_01288_01289_01290_and_visual_check"})
        if row["file_name"] in QUARANTINE:
            excluded.append({"row": row, "reason": "identical_pixels_with_conflicting_labels_user_requested_exclusion"})
            continue
        previous = seen.get(row["file_name"])
        if previous is not None:
            if previous != row:
                raise ValueError(f"Annotations contradictoires non resolues : {row['file_name']}")
            changes.append({"line": number, "file_name": row["file_name"], "reason": "identical_metadata_duplicate_removed"})
            continue
        seen[row["file_name"]] = row
        clean.append(row)
    return clean, changes, excluded


def stratified_validation(entries, ratio, seed):
    groups = defaultdict(list)
    for row in entries:
        groups[str(row["source"])].append(row)
    target = round(len(entries) * ratio)
    allocations = {key: min(len(rows) - 1, math.floor(len(rows) * ratio)) for key, rows in groups.items()}
    ranking = sorted(groups, key=lambda key: (-(len(groups[key]) * ratio % 1), key))
    while sum(allocations.values()) < target:
        candidates = [key for key in ranking if allocations[key] < len(groups[key]) - 1]
        if not candidates:
            raise ValueError("Effectif de validation incompatible avec les sources disponibles.")
        for key in candidates:
            if sum(allocations.values()) == target:
                break
            allocations[key] += 1
    train, validation = [], []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: row["file_name"])
        random.Random(f"{seed}:{key}").shuffle(rows)
        validation.extend(rows[:allocations[key]])
        train.extend(rows[allocations[key]:])
    return sorted(train, key=lambda r: r["file_name"]), sorted(validation, key=lambda r: r["file_name"])


def prepare(dataset, output, ratio=0.1, seed=42):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((dataset / "download_report.json").read_text(encoding="utf-8"))
    if receipt["revision"] != REVISION:
        raise ValueError("Revision du dataset differente : ne pas reutiliser ce protocole sans nouvel audit.")
    metadata = dataset / "data/train/metadata.jsonl"
    backups = sorted(metadata.parent.glob("metadata.original.*.jsonl"))
    if len(backups) > 1:
        raise ValueError("Plusieurs sauvegardes originales : selection manuelle necessaire.")
    if backups:
        original_path = backups[0]
    else:
        original_path = metadata.with_name(f"metadata.original.{sha256(metadata)[:12]}.jsonl")
        original_path.write_bytes(metadata.read_bytes())
    original = read_rows(original_path)
    clean, changes, excluded = corrected_rows(original)
    test_path = dataset / "data/test/metadata.jsonl"
    test_hash = sha256(test_path)
    public_test = read_rows(test_path)
    if len(public_test) != 400:
        raise ValueError("Le test public attendu contient exactement 400 exemples.")
    entries = {}
    for partition, rows in (("train", clean), ("test", public_test)):
        values = []
        for row in rows:
            image_path = dataset / "data" / partition / row["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            value = dict(row, partition=partition, transcript=unicodedata.normalize("NFC", row["transcript"]))
            value["pixel_sha256"] = pixel_hash(image_path)
            value["file_sha256"] = sha256(image_path)
            with Image.open(image_path) as image:
                value["width"], value["height"] = image.size
            values.append(value)
        entries[partition] = values
    test_pixels = {row["pixel_sha256"] for row in entries["test"]}
    if len(test_pixels) != len(public_test):
        raise ValueError("Doublon visuel dans le test public : il doit etre audite sans changer le test silencieusement.")
    seen_pixels, retained = {}, []
    for row in entries["train"]:
        key = row["pixel_sha256"]
        if key in test_pixels:
            excluded.append({"row": row, "reason": "exact_pixels_already_in_public_test"})
        elif key in seen_pixels:
            raise ValueError(f"Autre doublon visuel a examiner : {seen_pixels[key]} / {row['file_name']}")
        else:
            seen_pixels[key] = row["file_name"]
            retained.append(row)
    train, validation = stratified_validation(retained, ratio, seed)
    splits = {"train": train, "validation": validation, "test": entries["test"]}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not ({r["pixel_sha256"] for r in splits[a]} & {r["pixel_sha256"] for r in splits[b]})
    chars = sorted({character for row in train for character in row["transcript"]})
    vocab = {"<BLANK>": 0, "<UNK>": 1, **{character: i + 2 for i, character in enumerate(chars)}}
    summary = {
        "author": "IX-World", "dataset_repo": "IntelligenceResearchLab/Hausa", "dataset_revision": REVISION,
        "seed": seed, "validation_ratio_of_public_train": ratio, "public_test_unchanged": True,
        "original_train_metadata_sha256": sha256(original_path), "test_metadata_sha256": test_hash,
        "counts": {name: len(rows) for name, rows in splits.items()}, "corrections": changes,
        "excluded": excluded, "vocab_policy": "training_references_only_with_UNK; evaluation_references_remain_verbatim_NFC",
        "unseen_characters": {part: sorted({c for r in splits[part] for c in r["transcript"]} - set(chars)) for part in ("validation", "test")},
        "source_distribution": {part: dict(Counter(r["source"] for r in rows)) for part, rows in splits.items()},
        "exact_pixel_overlap": 0, "near_duplicates_checked": False, "source_held_out_validation": False,
    }
    infeasible = []
    for part, rows in splits.items():
        for row in rows:
            text = row["transcript"]
            minimum = len(text) + sum(a == b for a, b in zip(text, text[1:]))
            steps = max(4, round(row["width"] * 96 / row["height"])) // 4
            if minimum > steps:
                infeasible.append({"partition": part, "file_name": row["file_name"], "required": minimum, "available": steps})
    summary["ctc_infeasible_at_height_96"] = infeasible
    if any(r["partition"] == "train" for r in infeasible):
        raise ValueError(f"Images train incompatibles CTC : {infeasible}")
    text = "".join(json.dumps({key: row[key] for key in ("file_name", "transcript", "source")}, ensure_ascii=False) + "\n" for row in retained)
    new_bytes = text.encode("utf-8")
    if metadata.read_bytes() not in (original_path.read_bytes(), new_bytes):
        raise ValueError("Les metadonnees ont ete modifiees depuis la sauvegarde : arret pour conserver les changements.")
    temporary = metadata.with_suffix(".jsonl.tmp")
    temporary.write_bytes(new_bytes)
    temporary.replace(metadata)
    assert sha256(test_path) == test_hash
    split_path = output / "split.json"
    existing = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else None
    if existing is not None and existing != splits:
        raise ValueError("Ce dossier contient deja un split different. Utiliser un autre dossier de protocole.")
    write_json(split_path, splits)
    write_json(output / "vocab.json", vocab)
    summary["split_sha256"] = sha256(split_path)
    summary["vocab_sha256"] = sha256(output / "vocab.json")
    summary["corrected_train_metadata_sha256"] = sha256(metadata)
    write_json(output / "protocol.json", summary)
    write_json(output / "quarantine.json", excluded)
    print(json.dumps({"counts": summary["counts"], "split_sha256": summary["split_sha256"], "unseen_characters": summary["unseen_characters"], "ctc_infeasible": infeasible}, ensure_ascii=True, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IX-World : corriger Hausa et isoler le test public de 400 images")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent.parent / "Hausa_repo_2026-10-15")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "protocol_hausa_20261015")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 0.5:
        parser.error("La validation doit etre une fraction comprise entre 0 et 0.5 du train public.")
    prepare(args.dataset, args.output, args.validation_ratio, args.seed)
