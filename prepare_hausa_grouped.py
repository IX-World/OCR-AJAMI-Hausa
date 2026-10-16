"""IX-World: build complete provenance groups with optional public-test preservation."""
import argparse
import itertools
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from hausa_ocr.protocol import validate_partitions
from prepare_hausa import sha256, write_json


def group_validation(rows, ratio):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError("Au moins deux groupes hors test sont necessaires pour train et validation.")
    target = len(rows) * ratio
    candidates = (subset for size in range(1, len(names)) for subset in itertools.combinations(names, size))
    chosen = min(candidates, key=lambda subset: (abs(sum(len(groups[g]) for g in subset) - target), -len(subset), subset))
    validation_groups = set(chosen)
    return [r for r in rows if r["group_id"] not in validation_groups], [r for r in rows if r["group_id"] in validation_groups]


def full_corpus_split(rows, validation_target=200, test_target=400):
    groups=defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(groups)<3:
        raise ValueError("Trois groupes distincts sont necessaires.")
    def select(names,target,remaining):
        subsets=(s for n in range(1,len(names)-remaining+1) for s in itertools.combinations(sorted(names),n))
        return set(min(subsets,key=lambda s:(abs(sum(len(groups[g]) for g in s)-target),-len(s),s)))
    val_groups=select(set(groups),validation_target,2)
    test_groups=select(set(groups)-val_groups,test_target,1)
    return {"train":[r for r in rows if r["group_id"] not in val_groups|test_groups],"validation":[r for r in rows if r["group_id"] in val_groups],"test":[r for r in rows if r["group_id"] in test_groups]}


def prepare(dataset, base_protocol, output, all_images=False):
    base = json.loads((base_protocol / "protocol.json").read_text(encoding="utf-8"))
    for name, field in (("split.json", "split_sha256"), ("vocab.json", "vocab_sha256")):
        if sha256(base_protocol / name) != base[field]:
            raise ValueError("Le protocole initial a ete modifie.")
    for part, field in (("train", "corrected_train_metadata_sha256"), ("test", "test_metadata_sha256")):
        if sha256(dataset / "data" / part / "metadata.jsonl") != base[field]:
            raise ValueError("Metadonnees du corpus modifiees.")
    old = json.loads((base_protocol / "split.json").read_text(encoding="utf-8"))
    validate_partitions(old, base)
    sources = sorted({str(r["source"]) for rows in old.values() for r in rows}, key=int)
    mapping = {s: "sources_1_2" if s in ("1", "2") else f"source_{s}" for s in sources}
    rows = {part: [dict(row, group_id=mapping[str(row["source"])]) for row in entries] for part, entries in old.items()}
    for entries in rows.values():
        for row in entries:
            path = dataset / "data" / row["partition"] / row["file_name"]
            if sha256(path) != row["file_sha256"]:
                raise ValueError(f"Image modifiee : {path}")
    test_groups = {r["group_id"] for r in rows["test"]}
    public_train = sorted(rows["train"] + rows["validation"], key=lambda r: r["file_name"])
    excluded = [dict(row, exclusion_reason="provenance_group_present_in_public_test") for row in public_train if row["group_id"] in test_groups]
    eligible = [row for row in public_train if row["group_id"] not in test_groups]
    train, validation = group_validation(eligible, 0.1)
    splits = {"train": train, "validation": validation, "test": rows["test"]}
    if all_images:
        all_rows=sorted([r for part in rows.values() for r in part],key=lambda r:(r["partition"],r["file_name"]))
        splits=full_corpus_split(all_rows)
        for part,entries in splits.items():
            for row in entries:row["split"]=part
        train,validation=splits["train"],splits["validation"]
        excluded=[]
    chars = sorted({c for row in train for c in row["transcript"]})
    vocab = {"<BLANK>": 0, "<UNK>": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    provenance = {"source_to_group": mapping, "known_shared_scribes": [{"sources": ["1", "2"], "name": "Ibrahim Yaqubu Gusau", "evidence": "Dataset README.md, manuscript table, sources 1 and 2"}], "dataset_readme_sha256": sha256(dataset / "README.md"), "unknown_scribe_identity_sources": [s for s in sources if s not in ("1", "2")], "scope": "Source-disjoint and known-scribe-disjoint; unknown scribes cannot be certified distinct."}
    report = dict(base)
    report.update(protocol_id="hausa_grouped_20261016", separation_unit="source_and_known_scribe", source_to_group=mapping, counts={p:len(r) for p,r in splits.items()}, public_test_unchanged=True, source_held_out_validation=True, writer_disjointness="known_links_only", unknown_scribe_identities=True, validation_ratio_target=0.1, validation_ratio_of_eligible_train=len(validation)/len(eligible), validation_selection="whole groups; nearest target size; ties prefer more groups then lexical order; no scores used", eligible_public_train=len(eligible), excluded_public_train_count=len(excluded), source_distribution={p:dict(Counter(str(r["source"]) for r in entries)) for p,entries in splits.items()}, group_assignment={p:sorted({r["group_id"] for r in entries}) for p,entries in splits.items()}, unseen_characters={p:sorted({c for r in splits[p] for c in r["transcript"]}-set(chars)) for p in ("validation","test")}, base_split_sha256=base["split_sha256"])
    if all_images:
        report.update(protocol_id="hausa_all_grouped_20261016",public_test_unchanged=False,original_public_partition_retained_as_provenance=True,all_usable_images_used=True,eligible_examples=sum(len(r) for r in splits.values()),eligible_public_train=len(public_train),validation_target_examples=200,test_target_examples=400,validation_selection="whole groups closest to 200 validation examples; then whole remaining groups closest to 400 test examples; ties prefer more groups then lexical order; no scores used")
        for field in ("validation_ratio_target","validation_ratio_of_public_train","validation_ratio_of_eligible_train"):
            report.pop(field,None)
        old_ids={p:{(r["partition"],r["file_name"]) for r in entries} for p,entries in old.items()}
        report["overlap_with_initial_protocol"]={p:{previous:len({(r["partition"],r["file_name"]) for r in entries}&ids) for previous,ids in old_ids.items()} for p,entries in splits.items()}
    validate_partitions(splits, report, require_groups=True)
    infeasible = []
    for part, entries in splits.items():
        for row in entries:
            text = row["transcript"]
            required = len(text) + sum(a == b for a,b in zip(text,text[1:]))
            if max(4,round(row["width"]*96/row["height"]))//4 < required:
                infeasible.append({"partition":part,"file_name":row["file_name"]})
    report["ctc_infeasible_at_height_96"] = infeasible
    if not all_images:
        report["validation_ratio_of_public_train"] = len(validation)/len(public_train)
    if any(r["partition"] == "train" for r in infeasible):
        raise ValueError("Un exemple train ne permet pas l'alignement CTC.")
    if output.exists() and (output / "split.json").exists() and json.loads((output / "split.json").read_text(encoding="utf-8")) != splits:
        raise FileExistsError("Un autre split existe deja dans ce dossier.")
    output.mkdir(parents=True,exist_ok=True)
    for name,value in (("split.json",splits),("vocab.json",vocab),("source_groups.json",provenance),("excluded_public_train.json",excluded)):
        write_json(output/name,value)
    shutil.copy2(base_protocol/"quarantine.json",output/"quarantine.json")
    for part,entries in splits.items():
        (output/f"{part}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in entries),encoding="utf-8",newline="\n")
    for name,field in (("split.json","split_sha256"),("vocab.json","vocab_sha256"),("source_groups.json","source_groups_sha256"),("excluded_public_train.json","excluded_public_train_sha256")):
        report[field] = sha256(output/name)
    write_json(output/"protocol.json",report)
    print(json.dumps({"counts":report["counts"],"sources":report["source_distribution"],"excluded":len(excluded),"unseen_characters":report["unseen_characters"],"split_sha256":report["split_sha256"]},ensure_ascii=False,indent=2))
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--base-protocol",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--all-images",action="store_true")
    args=parser.parse_args()
    prepare(args.dataset.resolve(),args.base_protocol.resolve(),args.output.resolve(),args.all_images)
