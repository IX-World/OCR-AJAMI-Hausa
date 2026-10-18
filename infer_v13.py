"""IX-World: local V13 inference using the packaged training architecture."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Transcrire un segment Ajami avec V13, sans enseignant ni entrainement.")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Identifiant documentaire du contexte archive ; pas une langue detectee.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    root = args.model_dir.resolve()
    sys.path.insert(0, str(root / "source"))
    import torch
    from PIL import Image, ImageOps
    from hausa_ocr.data import decode
    from hausa_ocr.semantic import context_values, load_context
    from hausa_ocr.v13 import SemanticCRNN
    from prepare_hausa import sha256

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    selected = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
    if config["version"] != "v13" or sha256(root / "best_cer.pt") != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint V13 absent ou modifie.")
    for name, expected in config["source_sha256"].items():
        if sha256(root / "source" / name) != expected:
            raise ValueError(f"Code archive modifie : {name}")
    vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    if sha256(root / "vocab.json") != config["vocab_sha256"]:
        raise ValueError("Vocabulaire modifie.")
    split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    sources = {str(r["source"]) for rows in split.values() for r in rows}
    _, contexts = load_context(root / "context.json", sources)
    from hausa_ocr.semantic import digest
    if digest(contexts) != config["resolved_context_sha256"] or args.source not in contexts:
        raise ValueError("Contexte modifie ou source inconnue.")
    if args.threads < 1:
        raise ValueError("--threads doit etre positif.")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    # This is our own verified PyTorch training checkpoint, including RNG/config.
    checkpoint = torch.load(root / "best_cer.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"] != config or checkpoint["vocab"] != vocab:
        raise ValueError("Configuration du checkpoint incompatible.")
    model = SemanticCRNN(len(vocab), config["context_schema"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    with Image.open(args.image) as original:
        image = original.convert("L")
    width = max(4, round(image.width * 96 / image.height))
    image = ImageOps.mirror(image.resize((width, 96), Image.Resampling.LANCZOS))
    gray = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(96, width).float() / 255.0
    context = {key: value.to(device) for key, value in context_values([{"source": args.source}], contexts, config["context_schema"]).items()}
    with torch.inference_mode():
        logits = model((gray * 2 - 1)[None, None].to(device), torch.tensor([width]), context)
    print(decode(logits.argmax(-1)[0, :width // 4].cpu(), vocab), flush=True)


if __name__ == "__main__":
    main()
