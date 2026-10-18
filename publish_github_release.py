"""IX-World: publish a verified local model archive as a GitHub release."""
import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import httpx
from hausa_ocr.protocol import validate_partitions


@contextmanager
def github_client():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", capture_output=True, text=True, timeout=30, env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never"))
        credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        token = credentials.get("password")
    if not token:
        raise RuntimeError("Connexion GitHub absente. Configurer Git Credential Manager ou GH_TOKEN.")
    with httpx.Client(headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}, timeout=httpx.Timeout(600, connect=30)) as client:
        response = client.get("https://api.github.com/user")
        response.raise_for_status()
        if response.json()["login"] != "IX-World":
            raise ValueError("Cette publication doit etre effectuee par IX-World.")
        yield client


def publish(args):
    if args.repo != "IX-World/OCR-AJAMI-Hausa":
        raise ValueError("Depot GitHub non prevu pour cette publication.")
    if not args.asset.is_file() or args.asset.suffix != ".zip":
        raise ValueError("L'asset doit etre un package ZIP verifie.")
    with zipfile.ZipFile(args.asset) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        files = set(archive.namelist())
        if manifest["author"] != "IX-World" or manifest["version"] not in ("v9", "v10", "v11", "v12", "v13") or files != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Contenu ou attribution du package incorrect.")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Fichier modifie dans l'archive : {name}")
        protocol=json.loads(archive.read("protocol.json"))
        validate_partitions(json.loads(archive.read("split.json")),protocol,require_groups=True)
        if not protocol.get("all_usable_images_used",False):
            raise ValueError("Le protocole actuel utilise tout le corpus valide.")
    asset_paths = [args.asset] + list(getattr(args, "extra_asset", ()))
    if len({path.name for path in asset_paths}) != len(asset_paths):
        raise ValueError("Les noms des assets doivent etre distincts.")
    for path in asset_paths[1:]:
        if not path.is_file() or manifest["files"].get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Asset complementaire absent du package ou modifie : {path.name}")
    api = f"https://api.github.com/repos/{args.repo}"
    with github_client() as client:
        found = client.get(f"{api}/releases/tags/{quote(args.tag, safe='')}")
        if found.status_code == 404:
            response = client.post(f"{api}/releases", json={"tag_name": args.tag, "target_commitish": args.commit, "name": args.title, "body": args.body_file.read_text(encoding="utf-8"), "draft": False, "prerelease": False, "make_latest": "true"})
            response.raise_for_status()
            release = response.json()
        else:
            found.raise_for_status()
            release = found.json()
            if release["draft"] or release["prerelease"]:
                raise ValueError("Une release differente existe deja pour ce tag.")
        uploaded = []
        for path in asset_paths:
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            size = path.stat().st_size
            existing = next((asset for asset in release.get("assets", []) if asset["name"] == path.name), None)
            if existing:
                if existing.get("digest") != digest or existing["size"] != size:
                    raise ValueError("Un asset de contenu different existe deja ; aucun remplacement automatique.")
                asset = existing
            else:
                upload_url = release["upload_url"].split("{", 1)[0]
                with path.open("rb") as handle:
                    response = client.post(upload_url, params={"name": path.name, "label": digest}, headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)}, content=handle)
                response.raise_for_status()
                asset = response.json()
            if asset["size"] != size or asset.get("digest") != digest or asset["state"] != "uploaded":
                raise ValueError("Empreinte ou taille de l'asset distant incorrecte.")
            uploaded.append({"asset_url": asset["browser_download_url"], "asset_sha256": digest.removeprefix("sha256:"), "asset_bytes": size})
        if getattr(args, "update_notes", False):
            response = client.patch(f"{api}/releases/{release['id']}", json={"body": args.body_file.read_text(encoding="utf-8")})
            response.raise_for_status()
            release = response.json()
        receipt = {"author": "IX-World", "repository": args.repo, "release_url": release["html_url"], "tag": args.tag, "target_commit": args.commit, **uploaded[0]}
        if uploaded[1:]:
            receipt["additional_assets"] = uploaded[1:]
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="IX-World/OCR-AJAMI-Hausa")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--body-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--extra-asset", type=Path, action="append", default=[], help="Fichier deja present dans le manifeste de l'archive, a exposer aussi en telechargement direct.")
    parser.add_argument("--update-notes", action="store_true", help="Mettre a jour la description de la release apres verification de tous les assets.")
    publish(parser.parse_args())
