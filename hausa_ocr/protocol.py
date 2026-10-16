"""Partition integrity checks shared by preparation, training and publication."""
from itertools import combinations


def validate_partitions(splits, report, require_groups=False):
    strict = report.get("separation_unit") == "source_and_known_scribe"
    if require_groups and not strict:
        raise ValueError("Ce lancement exige le nouveau protocole separe par manuscrit et copiste identifie.")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Trois partitions train/validation/test sont requises.")
    mapping = report.get("source_to_group", {})
    if strict and "1" in mapping and "2" in mapping and mapping["1"] != mapping["2"]:
        raise ValueError("Les sources 1 et 2 partagent un copiste documente.")
    values = {}
    for part, rows in splits.items():
        ids = {(r["partition"], r["file_name"]) for r in rows}
        pixels = {r["pixel_sha256"] for r in rows}
        if not rows or len(ids) != len(rows) or len(pixels) != len(rows):
            raise ValueError(f"Partition {part} vide ou contenant des doublons.")
        origin = "test" if part == "test" else "train"
        if any(r["partition"] != origin for r in rows):
            raise ValueError("Origine publique train/test incorrecte.")
        sources = {str(r["source"]) for r in rows}
        if strict and any(mapping.get(str(r["source"])) != r.get("group_id") for r in rows):
            raise ValueError("Groupe de provenance incoherent.")
        values[part] = {"ids": ids, "pixels": pixels, "sources": sources, "groups": {mapping[s] for s in sources} if strict else set()}
    for a, b in combinations(splits, 2):
        for key in ("ids", "pixels") + (("sources", "groups") if strict else ()):
            if values[a][key] & values[b][key]:
                raise ValueError(f"Chevauchement {key} entre {a} et {b}.")
    return strict
