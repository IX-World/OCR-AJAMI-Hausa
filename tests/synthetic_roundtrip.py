"""Exercise train, resume and test on six synthetic images, never on corpus data."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import train_hausa
from prepare_hausa import sha256, write_json
from publish_hausa import package_run

temporary = tempfile.TemporaryDirectory(prefix="synthetic-", dir=root / "tests")
assert Path(temporary.name).resolve().is_relative_to((root / "tests").resolve())
with temporary as directory:
    base = Path(directory)
    dataset, protocol, run = base / "dataset", base / "protocol", base / "run"
    split = {}
    for part in ("train", "validation", "test"):
        split[part] = []
        folder = dataset / "data" / ("test" if part == "test" else "train")
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            name = f"{part}_{i}.png"
            image = Image.new("L", (32, 96), 220 - i * 15)
            image.putpixel((i, 0), len(part))
            image.save(folder / name)
            split[part].append({"partition": folder.name, "file_name": name, "source": "synthetic", "transcript": "a", "width": 32, "height": 96, "file_sha256": sha256(folder / name)})
    for part in ("train", "test"):
        (dataset / "data" / part / "metadata.jsonl").write_text("synthetic fixture only\n", encoding="utf-8")
    write_json(protocol / "split.json", split)
    write_json(protocol / "vocab.json", {"<BLANK>": 0, "<UNK>": 1, "a": 2})
    write_json(protocol / "quarantine.json", [])
    write_json(protocol / "protocol.json", {"dataset_repo": "synthetic-test-not-a-real-dataset", "dataset_revision": "synthetic-test-not-publishable", "counts": {part: 2 for part in split}, "split_sha256": sha256(protocol / "split.json"), "vocab_sha256": sha256(protocol / "vocab.json"), "corrected_train_metadata_sha256": sha256(dataset / "data/train/metadata.jsonl"), "test_metadata_sha256": sha256(dataset / "data/test/metadata.jsonl"), "vocab_policy": "training_only", "unseen_characters": {"validation": [], "test": []}, "seed": 42})
    common = ["train_hausa.py", "--version", "v9", "--dataset", str(dataset), "--protocol", str(protocol), "--output", str(run), "--batch-size", "2", "--min-epochs", "1", "--patience", "2"]
    with patch("publish_hausa.write_model_card"), contextlib.redirect_stdout(io.StringIO()):
        with patch.object(sys, "argv", common + ["--epochs", "1"]):
            train_hausa.main()
        with patch.object(sys, "argv", common + ["--resume", "--epochs", "2"]):
            train_hausa.main()
        before = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
        with patch.object(sys, "argv", ["train_hausa.py", "--version", "v9", "--dataset", str(dataset), "--protocol", str(protocol), "--output", str(run), "--test-only"]):
            train_hausa.main()
    after = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    assert before["cer"] == after["cer"] and before["wer"] == after["wer"]
    assert len((run / "history.csv").read_text(encoding="utf-8").splitlines()) == 3
    try:
        package_run(run, base / "must-not-publish")
    except ValueError:
        pass
    else:
        raise AssertionError("Synthetic results must not be publishable as test400 results")
print("Synthetic train(1 epoch), resume(1 epoch), test-only and publication rejection: OK")
