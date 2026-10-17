"""IX-World: prepare frozen contextual AfriBERTa targets using TRAIN only."""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from hausa_ocr.semantic import (
    TEACHER_REPO, TEACHER_REVISION, TEACHER_WEIGHTS_SHA256, coverage, digest,
    fit_context_schema, load_annotations, load_context, require_frozen_split, sample_id, text_input,
)
from hausa_ocr.protocol import validate_partitions
from prepare_hausa import sha256, write_json


def dependencies(directory, install=False):
    requirements = Path(__file__).with_name("requirements-v13-semantic.txt")
    expected = dict(line.strip().split("==") for line in requirements.read_text().splitlines() if line.strip() and not line.startswith("#"))
    found = {d.metadata["Name"].lower().replace("-", "_"): d.version for d in importlib.metadata.distributions(path=[str(directory)])}
    ready = all(found.get(name.lower().replace("-", "_")) == version for name, version in expected.items())
    if not ready:
        if not install:
            raise RuntimeError("Dependances semantiques absentes. Executer preparer_v13.cmd ou ajouter --install-deps.")
        print(f"Installation isolee des dependances dans {directory}; l'environnement OCR est conserve.", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "--target", str(directory), "-r", str(requirements)], check=True)
    sys.path.insert(0, str(directory))
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def load_training_manifest(protocol):
    report = json.loads((protocol / "protocol.json").read_text(encoding="utf-8"))
    require_frozen_split(report)
    for name, field in (("split.json", "split_sha256"), ("vocab.json", "vocab_sha256"), ("source_groups.json", "source_groups_sha256")):
        if sha256(protocol / name) != report[field]:
            raise ValueError(f"Protocole modifie : {name}")
    splits = json.loads((protocol / "split.json").read_text(encoding="utf-8"))
    validate_partitions(splits, report, require_groups=True)
    # Only IDs/source metadata of held-out rows are used for integrity checks.
    # No image, validation/test transcript, prediction or metric enters the teacher.
    sources = set(report["source_to_group"])
    return splits["train"], sources, report


def load_teacher(model_cache):
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaTokenizerFast
    from transformers.convert_slow_tokenizer import XLMRobertaConverter
    names = ["config.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "pytorch_model.bin"]
    files = {name: Path(hf_hub_download(TEACHER_REPO, name, revision=TEACHER_REVISION, cache_dir=str(model_cache))) for name in names}
    if sha256(files["pytorch_model.bin"]) != TEACHER_WEIGHTS_SHA256:
        raise ValueError("Empreinte des poids AfriBERTa incorrecte.")
    if sha256(files["sentencepiece.bpe.model"]) != "6419b3044bff45e94e0553cbb81425fd06046e9294b33555e23fdc69377dba6f":
        raise ValueError("Tokenizer AfriBERTa modifie.")
    folder = files["config.json"].parent
    config = AutoConfig.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
    config._attn_implementation = "eager"
    model = AutoModel.from_config(config, add_pooling_layer=False, trust_remote_code=False)
    # AfriBERTa currently supplies a .bin MLM checkpoint. Load only tensors,
    # never pickle arbitrary objects and never disable Transformers safety checks.
    state = torch.load(files["pytorch_model.bin"], map_location="cpu", weights_only=True)
    state = {key.removeprefix("roberta."): value for key, value in state.items() if key.startswith("roberta.")}
    legacy_positions = state.pop("embeddings.position_ids", None)
    if legacy_positions is not None and not torch.equal(legacy_positions, model.embeddings.position_ids):
        raise ValueError("Buffer de positions historique AfriBERTa incompatible.")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    slow_tokenizer = AutoTokenizer.from_pretrained(str(folder), use_fast=False, local_files_only=True, trust_remote_code=False)
    class ByteFallbackConverter(XLMRobertaConverter):
        handle_byte_fallback = True
    tokenizer = XLMRobertaTokenizerFast(tokenizer_object=ByteFallbackConverter(slow_tokenizer).converted())
    tokenizer.v13_reference_tokenizer = slow_tokenizer
    if not tokenizer.is_fast:
        raise RuntimeError("Un tokenizer avec offsets est requis pour isoler la transcription dans son contexte.")
    tokenizer.model_max_length = 512
    return model, tokenizer, {name: sha256(path) for name, path in files.items()}


@torch.inference_mode()
def encode_targets(model, tokenizer, requests, batch_size=8, max_length=256):
    output = []
    for start in range(0, len(requests), batch_size):
        group = requests[start:start + batch_size]
        batch = tokenizer([r["text"] for r in group], padding=True, truncation=False, return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors="pt")
        reference_tokenizer = getattr(tokenizer, "v13_reference_tokenizer", None)
        if reference_tokenizer is not None:
            original = reference_tokenizer([r["text"] for r in group], padding=True, truncation=False, return_tensors="pt")
            if not torch.equal(original["input_ids"], batch["input_ids"]):
                raise ValueError("Tokenisation differente du SentencePiece original : ne pas supprimer les diacritiques pour contourner cette erreur.")
        if batch["input_ids"].shape[1] > max_length:
            raise ValueError("Contexte trop long : reduire l'annotation ou augmenter --max-length (maximum 512). Aucune troncature silencieuse.")
        offsets = batch.pop("offset_mapping")
        special = batch.pop("special_tokens_mask")
        hidden = model(**batch).last_hidden_state
        mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
        for i, request in enumerate(group):
            mask[i] = (offsets[i, :, 1] > request["start"]) & (offsets[i, :, 0] < request["end"]) & (special[i] == 0) & batch["attention_mask"][i].bool()
        if not bool(mask.any(1).all()):
            raise ValueError("Transcription vide ou non tokenisable pour la supervision semantique.")
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        output.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
        if start == 0 or (start // batch_size + 1) % 25 == 0 or start + len(group) == len(requests):
            print(f"Encodeur CPU : {start + len(group)}/{len(requests)} textes TRAIN uniques", flush=True)
    return torch.cat(output)


def main():
    parser = argparse.ArgumentParser(description="Preparer les cibles semantiques TRAIN de V13, sans entrainer l'OCR.")
    base = Path(__file__).resolve().parent.parent
    parser.add_argument("--protocol", type=Path, default=base / "protocol_hausa_all_grouped_20261016")
    default_context = base / "v13_context.json"
    parser.add_argument("--context", type=Path, default=default_context if default_context.exists() else Path(__file__).with_name("configs") / "v13_context.json")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, default=base / "v13_semantics")
    parser.add_argument("--model-cache", type=Path, default=base / "v13_encoder_cache")
    parser.add_argument("--deps", type=Path, default=Path(__file__).with_name(".v13-semantic-deps"))
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or not 8 <= args.max_length <= 512:
        parser.error("Batch/threads positifs ; max-length entre 8 et 512.")
    torch.set_num_threads(args.threads)
    rows, sources, report = load_training_manifest(args.protocol)
    raw, contexts = load_context(args.context, sources)
    annotations = load_annotations(args.annotations, rows)
    schema = fit_context_schema(rows, contexts)
    train_contexts = {str(row["source"]): contexts[str(row["source"])] for row in rows}
    identity = {"schema_version": 1, "split_sha256": report["split_sha256"], "vocab_sha256": report["vocab_sha256"],
                "train_rows_sha256": digest(rows), "train_contexts": train_contexts, "context_schema": schema,
                "annotations_sha256": digest(annotations), "teacher_repo": TEACHER_REPO, "teacher_revision": TEACHER_REVISION,
                "teacher_weights_sha256": TEACHER_WEIGHTS_SHA256, "pooling": "mean_reference_span_excluding_special_padding_tokens",
                "tokenizer_policy": "byte_fallback_enabled_verified_against_original_sentencepiece_for_every_input",
                "max_length": args.max_length, "preparation_source_sha256": sha256(Path(__file__)),
                "semantic_source_sha256": sha256(Path(__file__).with_name("hausa_ocr") / "semantic.py")}
    fingerprint = digest(identity)
    target_path = args.output / "train_targets.pt"
    if target_path.exists():
        manifest_path = args.output / "manifest.json"
        if not manifest_path.exists() or not (args.output / "annotations.json").exists():
            raise FileExistsError("Cache incomplet : conserver ce dossier et choisir un nouvel --output.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_sha256") != sha256(target_path):
            raise ValueError("Empreinte du cache semantique incorrecte.")
        cached = torch.load(target_path, map_location="cpu", weights_only=True)
        if cached.get("identity_sha256") != fingerprint:
            raise FileExistsError("Un cache semantique different existe ; choisir un nouvel --output pour conserver la tracabilite.")
        if cached.get("sample_ids") != [sample_id(row) for row in rows] or cached["targets"].shape != (len(rows), 768) or not torch.isfinite(cached["targets"]).all():
            raise ValueError("Cache semantique invalide.")
        print(f"Cache TRAIN deja prepare : {target_path}", flush=True)
        return
    dependencies(args.deps, args.install_deps)
    model, tokenizer, teacher_files = load_teacher(args.model_cache)
    requests, request_index, assignments = [], {}, []
    for row in rows:
        language = contexts[str(row["source"])]["language"]
        if language is None:
            assignments.append(None)
            continue
        text, start, end = text_input(row, annotations.get(sample_id(row)))
        request = dict(text=text, start=start, end=end, language=language)
        key = digest(request)
        if key not in request_index:
            request_index[key] = len(requests)
            requests.append(request)
        assignments.append(request_index[key])
    if not requests:
        raise ValueError("Aucune langue de train documentee : aucune supervision semantique ne peut etre preparee.")
    vectors = encode_targets(model, tokenizer, requests, args.batch_size, args.max_length)
    if vectors.shape[1] != 768 or not torch.isfinite(vectors).all():
        raise ValueError("Cibles semantiques invalides.")
    targets = torch.stack([vectors[i] if i is not None else torch.zeros(768) for i in assignments])
    positive_keys = [digest(requests[i]) if i is not None else "unknown:" + sample_id(row) for row, i in zip(rows, assignments)]
    payload = {"identity": identity, "identity_sha256": fingerprint, "sample_ids": [sample_id(row) for row in rows], "positive_keys": positive_keys,
               "targets": targets, "context_schema": schema, "teacher_files_sha256": teacher_files,
               "training_only": True, "teacher_frozen": True, "teacher_device": "cpu", "torch": str(torch.__version__),
               "transformers": importlib.metadata.version("transformers"), "coverage_train": coverage(rows, contexts),
               "verified_contextual_annotations": len(annotations), "unique_teacher_inputs": len(requests)}
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(target_path)
    manifest = {key: value for key, value in payload.items() if key not in ("targets", "sample_ids")}
    manifest["cache_sha256"] = sha256(target_path)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "context.json", raw)
    write_json(args.output / "annotations.json", annotations)
    print(f"Cibles preparees : {target_path} | {len(rows)} TRAIN | validation/test exclus", flush=True)
    print(f"Couverture : {payload['coverage_train']} | contextes textuels verifies : {len(annotations)}", flush=True)
    if payload["coverage_train"]["known_period"] == 0 or payload["coverage_train"]["known_locality"] == 0:
        print("Epoque/localite absentes : leurs entrees sont masquees. Aucun apprentissage diachronique/regional n'est affirme.", flush=True)


if __name__ == "__main__":
    main()
