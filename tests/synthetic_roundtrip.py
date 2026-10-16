"""Exercise train, resume and test on six synthetic images, never on corpus data."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import train_hausa
from prepare_hausa import sha256, write_json
from publish_hausa import load_complete_metrics, package_run, write_model_card

torch.set_num_threads(2)

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
    common = ["train_hausa.py", "--version", "v9", "--dataset", str(dataset), "--protocol", str(protocol), "--output", str(run), "--batch-size", "2", "--min-epochs", "1", "--patience", "2", "--device", "cpu"]
    with patch("publish_hausa.write_model_card"), contextlib.redirect_stdout(io.StringIO()):
        with patch.object(sys, "argv", common + ["--fresh-start", "--epochs", "1"]):
            train_hausa.main()
        assert json.loads((run / "config.json").read_text(encoding="utf-8"))["fresh_start_required"]
        with patch.object(sys, "argv", common + ["--resume", "--epochs", "2"]):
            train_hausa.main()
        before = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
        checkpoint_before = sha256(run / "best_cer.pt")
        with patch.object(sys, "argv", ["train_hausa.py", "--version", "v9", "--dataset", str(dataset), "--protocol", str(protocol), "--output", str(run), "--evaluate-only", "--device", "cpu"]):
            train_hausa.main()
            with patch.object(train_hausa, "__file__", str(run / "evaluation_source" / "train_hausa.py")):
                train_hausa.main()
    after = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    assert before["cer"] == after["cer"] and before["wer"] == after["wer"]
    assert checkpoint_before == sha256(run / "best_cer.pt")
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for part, values in summary["partitions"].items():
        assert values["examples"] == 2 and values["augmented"] is False
        assert all(field in values for field in ("cer", "wer", "uwer"))
        assert values["checkpoint_sha256"] == checkpoint_before
        assert len((run / f"{part}_predictions.csv").read_text(encoding="utf-8-sig").splitlines()) == 3
    assert len((run / "metrics.csv").read_text(encoding="utf-8-sig").splitlines()) == 4
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert load_complete_metrics(run, config, after) == summary
    for name, expected in config["source_sha256"].items():
        assert sha256(run / "source" / name) == expected
    with patch("publish_hausa.load_run", return_value=(config, after, json.loads((protocol / "protocol.json").read_text(encoding="utf-8")))):
        write_model_card(run)
    card = (run / "README.md").read_text(encoding="utf-8")
    assert "| train | 2 |" in card and "| validation | 2 |" in card and "| test | 2 |" in card
    assert "UWER" in card and "casefold" in card
    assert len((run / "history.csv").read_text(encoding="utf-8").splitlines()) == 3
    initial_sha = sha256(run / "best_cer.pt")
    fine = base / "finetuned"
    with patch("publish_hausa.write_model_card"), contextlib.redirect_stdout(io.StringIO()):
        with patch.object(sys, "argv", common + ["--output", str(fine), "--finetune-from", str(run / "best_cer.pt"), "--learning-rate", "0.0001", "--epochs", "1"]):
            train_hausa.main()
        try:
            with patch.object(sys, "argv", common + ["--version", "v10", "--output", str(base / "wrong-version"), "--finetune-from", str(run / "best_cer.pt"), "--epochs", "1"]):
                train_hausa.main()
        except ValueError:
            pass
        else:
            raise AssertionError("A different model version cannot initialize fine tuning")
    fine_config = json.loads((fine / "config.json").read_text(encoding="utf-8"))
    assert not fine_config["from_scratch"] and fine_config["learning_rate"] == 0.0001
    assert fine_config["initial_checkpoint_sha256"] == initial_sha == sha256(run / "best_cer.pt")
    try:
        package_run(run, base / "must-not-publish")
    except ValueError:
        pass
    else:
        raise AssertionError("Synthetic results must not be publishable as test400 results")
print("Synthetic train, resume, test-only, fine tuning, version guard and publication rejection: OK")
