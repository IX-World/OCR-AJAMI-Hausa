import csv
import hashlib
import itertools
import json
import math
import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from hausa_ocr.ctc_beam import prefix_beam_search, tokens_to_text


def collapse(path, blank=0):
    return tuple(value for i, value in enumerate(path) if value != blank and (i == 0 or value != path[i-1]))


class BeamTests(unittest.TestCase):
    def test_matches_exhaustive_alignment_probabilities(self):
        rng = random.Random(42)
        for steps in (1, 2, 3, 4):
            probabilities = [[rng.random() for _ in range(3)] for _ in range(steps)]
            probabilities = [[p / sum(row) for p in row] for row in probabilities]
            exact = defaultdict(float)
            for path in itertools.product(range(3), repeat=steps):
                exact[collapse(path)] += math.prod(probabilities[t][label] for t, label in enumerate(path))
            actual = dict(prefix_beam_search([[math.log(p) for p in row] for row in probabilities], beam_width=128, nbest=128))
            self.assertEqual(set(actual), set(exact))
            for tokens in exact:
                self.assertAlmostEqual(math.exp(actual[tokens]), exact[tokens], places=12)

    def test_sums_paths_instead_of_selecting_greedy_path(self):
        probabilities = [[math.log(0.6), math.log(0.4)]] * 3
        self.assertEqual(prefix_beam_search(probabilities, beam_width=10)[0][0], (1,))

    def test_repeated_letters_require_blank_and_are_not_collapsed_twice(self):
        probabilities = [[-math.inf, 0.0], [0.0, -math.inf], [-math.inf, 0.0]]
        tokens, log_probability = prefix_beam_search(probabilities)[0]
        self.assertEqual(tokens, (1, 1))
        self.assertEqual(log_probability, 0.0)
        self.assertEqual(tokens_to_text(tokens, {"<BLANK>": 0, "a": 1}), "aa")
        self.assertEqual(tokens_to_text((1, 2, 3), {"<BLANK>": 0, "<UNK>": 1, " ": 2, "ɗ": 3}), "\ufffd ɗ")

    def test_empty_input_ties_pruning_and_invalid_values(self):
        self.assertEqual(prefix_beam_search([]), [((), 0.0)])
        probabilities = [[math.log(0.3), math.log(0.4), math.log(0.3)]] * 3
        self.assertEqual(prefix_beam_search(probabilities), prefix_beam_search(probabilities, token_top_k=99))
        for bad in ([[math.nan, -1]], [[-math.inf, -math.inf]], [[0.5, -1]], [[0, -1], [0]]):
            with self.assertRaises(ValueError):
                prefix_beam_search(bad)

    def test_validation_loader_never_opens_heldout_data_or_metrics(self):
        from eval_beam_v10_validation import load_validation, score
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            dataset = root / "dataset"
            image = dataset / "data" / "test" / "validation.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"validation-only fixture")
            row = dict(partition="test", split="validation", file_name="validation.png", transcript="ɗā", source="15", group_id="g15", file_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
            split = {"train": [{"file_name": "train-never-open.png"}], "validation": [row], "test": [{"file_name": "heldout-never-open.png"}]}
            def save(name, value):
                path = run / name
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                return hashlib.sha256(path.read_bytes()).hexdigest()
            split_sha = save("split.json", split)
            vocab_sha = save("vocab.json", {"<BLANK>": 0, "<UNK>": 1, "ɗ": 2, "ā": 3})
            save("protocol.json", dict(split_sha256=split_sha, vocab_sha256=vocab_sha, separation_unit="source_and_known_scribe", all_usable_images_used=True, source_to_group={"15": "g15"}))
            save("config.json", dict(version="v10", channels=1, counts={"validation": 1}, split_sha256=split_sha, vocab_sha256=vocab_sha, dataset=str(dataset)))
            save("validation_metrics.json", dict(evaluated_partition="validation", augmented=False, examples=1, split_sha256=split_sha))
            with (run / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["partition", "file_name", "reference", "prediction"])
                writer.writeheader()
                writer.writerow(dict(partition="test", file_name="validation.png", reference="ɗā", prediction="ɗā"))
            native_open = Path.open
            def guarded_open(path, *args, **kwargs):
                if path.name in ("heldout-never-open.png", "train-never-open.png", "test_metrics.json", "test_predictions.csv"):
                    self.fail(f"Acces interdit : {path}")
                return native_open(path, *args, **kwargs)
            with patch.object(Path, "open", guarded_open):
                _, rows, _, _, _, _ = load_validation(run)
                self.assertEqual(rows, [row])
                self.assertEqual(score(["ɗā"], rows)["wer"], 0)
                with self.assertRaises(ValueError):
                    score([], rows)
            row["split"] = "test"
            split_sha = save("split.json", split)
            config = json.loads((run / "config.json").read_text())
            protocol = json.loads((run / "protocol.json").read_text())
            config["split_sha256"] = protocol["split_sha256"] = split_sha
            save("config.json", config)
            save("protocol.json", protocol)
            with self.assertRaises(ValueError):
                load_validation(run)


if __name__ == "__main__":
    unittest.main()
