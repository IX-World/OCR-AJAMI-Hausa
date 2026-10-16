import json
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hausa_ocr.data import WidthSampler, collate, decode, edit_distance
from hausa_ocr.models import build_model
from prepare_hausa import corrected_rows, stratified_validation
from publish_hausa import load_run


class ProtocolTests(unittest.TestCase):
    def test_repairs_keep_transcriptions_and_reject_unknown_conflicts(self):
        rows = [
            {"file_name": "ajami_00462png", "transcript": "mai", "source": "17"},
            {"file_name": "ajami_01288.png", "transcript": "a", "source": "9"},
            {"file_name": "ajami_01299.png", "transcript": "ranar", "source": "9"},
            {"file_name": "ajami_01290.png", "transcript": "illakhira", "source": "9"},
            {"file_name": "ajami_01299.png", "transcript": "arzuqni", "source": "9"},
            {"file_name": "duplicate.png", "transcript": "same", "source": "8"},
            {"file_name": "duplicate.png", "transcript": "same", "source": "8"},
            {"file_name": "a_ajami_03400.png", "transcript": "da", "source": "8"},
            {"file_name": "a_ajami_03401.png", "transcript": "mutane", "source": "8"},
        ]
        fixed, changes, excluded = corrected_rows(rows)
        references = {row["file_name"]: row["transcript"] for row in fixed}
        self.assertEqual(references["ajami_00462.png"], "mai")
        self.assertEqual(references["ajami_01289.png"], "ranar")
        self.assertEqual(references["ajami_01299.png"], "arzuqni")
        self.assertEqual(len(excluded), 2)
        self.assertEqual(sum(row["file_name"] == "duplicate.png" for row in fixed), 1)
        self.assertEqual(rows[0]["file_name"], "ajami_00462png")
        with self.assertRaises(ValueError):
            corrected_rows([dict(rows[5]), dict(rows[5], transcript="other")])

    def test_validation_is_deterministic_disjoint_and_uses_only_supplied_train(self):
        rows = [{"source": str(i % 5), "file_name": f"{i:03}.png", "transcript": "x"} for i in range(101)]
        train, validation = stratified_validation(rows, 0.1, 42)
        self.assertEqual(len(validation), 10)
        self.assertEqual((train, validation), stratified_validation(list(reversed(rows)), 0.1, 42))
        self.assertFalse({r["file_name"] for r in train} & {r["file_name"] for r in validation})
        self.assertEqual(len(train) + len(validation), len(rows))
        self.assertNotEqual(validation, stratified_validation(rows, 0.1, 43)[1])

    def test_padding_and_ctc_repeated_characters(self):
        row = {"file_name": "a.png"}
        result = collate([(torch.zeros(2, 96, 4), torch.tensor([2]), row, 4), (torch.zeros(2, 96, 8), torch.tensor([2]), row, 8)])
        self.assertTrue(torch.all(result["images"][0, 0, :, 4:] == 1))
        self.assertTrue(torch.all(result["images"][0, 1, :, 4:] == -1))
        vocab = {"<BLANK>": 0, "<UNK>": 1, "a": 2}
        self.assertEqual(decode([0, 2, 2, 0, 2, 0], vocab), "aa")
        self.assertEqual(decode([1, 0], vocab), "\ufffd")
        self.assertEqual(edit_distance("A b", "a b"), 1)
        self.assertEqual(edit_distance(["a", "b"], ["ab"]), 2)

    def test_sampler_does_not_drop_examples_and_reports_correct_batch_count(self):
        class Toy:
            expected_widths = list(reversed(range(403)))

            def __len__(self):
                return len(self.expected_widths)
        sampler = WidthSampler(Toy(), 20, 42)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(403)))
        self.assertLessEqual(max(map(len, batches)), 20)

    def test_architectures_keep_historical_parameter_counts(self):
        expected = {"v9": 6108625, "v10": 6108625, "v11": 6109201, "v12": 7622353}
        for version, count in expected.items():
            model = build_model(version, 81)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)

    def test_missing_real_training_results_cannot_be_published(self):
        missing = Path(__file__).resolve().parent / "a_run_that_has_never_been_trained"
        self.assertFalse(missing.exists())
        with self.assertRaises(FileNotFoundError):
            load_run(missing)

    def test_fresh_start_refuses_checkpoint_initialization(self):
        import train_hausa
        for flags in (["--resume"], ["--finetune-from", "missing.pt"]):
            with patch.object(sys, "argv", ["train_hausa.py", "--version", "v9", "--fresh-start"] + flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    train_hausa.main()
            self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
