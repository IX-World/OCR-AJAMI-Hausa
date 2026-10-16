import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CACHE", str(destination.parent / ".hf_xet_cache"))

    from huggingface_hub import HfApi, snapshot_download

    repo = "IntelligenceResearchLab/Hausa"
    api = HfApi()
    info = api.dataset_info(repo, revision=args.revision, files_metadata=True)
    sha = info.sha
    print(f"Dataset : {repo}\nRevision figee : {sha}", flush=True)
    print(f"Destination : {destination}", flush=True)
    known_size = sum(f.size or 0 for f in info.siblings)
    print(f"Taille des fichiers du depot : {known_size / 1024**2:.1f} MiB", flush=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=sha,
        local_dir=str(destination),
        max_workers=4,
    )

    report = {
        "repo_id": repo,
        "revision": sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination": str(destination),
        "splits": {},
    }
    for split in ("train", "test"):
        folder = destination / "data" / split
        metadata = folder / "metadata.jsonl"
        rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        names = [row["file_name"] for row in rows]
        counts = Counter(names)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        missing = sorted({name for name in names if not (folder / name).is_file()})
        referenced = {name for name in names if (folder / name).is_file()}
        images = {p.name for p in folder.iterdir() if p.suffix.lower() == ".png"}
        details = {
            "metadata_rows": len(rows),
            "png_images": len(images),
            "valid_unique_references": len(referenced),
            "missing_paths": missing,
            "duplicate_file_names": duplicates,
            "unreferenced_images": sorted(images - referenced),
        }
        report["splits"][split] = details
        print(f"{split.upper()} : {len(images)} PNG, {len(rows)} entrees, {len(referenced)} images referencees distinctes", flush=True)
        if missing or duplicates:
            print(f"  Metadonnees : {len(missing)} chemins absents, {len(duplicates)} noms repetes. Voir le rapport.", flush=True)

    expected = {f.rfilename for f in info.siblings}
    absent = [name for name in sorted(expected) if not (destination / name).is_file()]
    report["repository_file_count"] = len(expected)
    report["missing_repository_files"] = absent
    report_path = destination / "download_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if absent:
        raise RuntimeError(f"Telechargement incomplet : {len(absent)} fichiers absents. Relancer la meme commande.")
    print(f"Telechargement complet : {len(expected)} fichiers verifies presents.", flush=True)
    print(f"Rapport : {report_path}", flush=True)
    print("Les partitions publiques sont conservees. Aucun entrainement lance.", flush=True)


if __name__ == "__main__":
    main()
