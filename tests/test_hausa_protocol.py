import json
import contextlib
import io
import sys
import unittest
import hashlib
import tempfile
import zipfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

from hausa_ocr.data import WidthSampler, collate, decode, edit_distance
from hausa_ocr.models import build_model
from hausa_ocr.metrics import normalize_uwer
from hausa_ocr.reporting import cer_trend, print_epoch_report
from hausa_ocr.protocol import validate_partitions
from prepare_hausa_grouped import group_validation
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

    def test_uwer_removes_only_case_and_decomposable_diacritics(self):
        self.assertEqual(normalize_uwer("ÁIKŌ A\u0301Iko"), "aiko aiko")
        self.assertEqual(normalize_uwer("Ƙ Ɓ Ɗ ' [ ]"), "ƙ ɓ ɗ ' [ ]")
        import train_hausa
        vocab = {"<BLANK>": 0, "<UNK>": 1, "a": 2}
        class FixedModel(torch.nn.Module):
            def forward(self, images):
                return torch.tensor([[[0.0, 0.0, 1.0]]])
        batch = {"images": torch.zeros(1, 1, 96, 4), "widths": torch.tensor([4]), "rows": [{"partition": "test", "file_name": "toy.png", "source": "toy", "transcript": "Á"}]}
        scores, predictions = train_hausa.evaluate(FixedModel(), [batch], vocab, torch.device("cpu"))
        self.assertEqual(scores["cer"], 1.0)
        self.assertEqual(scores["wer"], 1.0)
        self.assertEqual(scores["uwer"], 0.0)
        self.assertEqual(scores["normalized_reference_words"], 1)
        self.assertEqual(predictions[0]["normalized_word_errors"], 0)

    def test_epoch_report_uses_validation_history_and_five_epoch_trend(self):
        records = [{"epoch": i, "val_cer": 0.7 - i * 0.05} for i in range(1, 7)]
        slope, gain = cer_trend(records)
        self.assertAlmostEqual(slope, -0.05)
        self.assertAlmostEqual(gain, 0.2)
        self.assertEqual(cer_trend(records[:1]), (None, None))
        config = {"version": "v12", "max_epochs": 100, "channels": 2, "cnn_convolutions": 9, "counts": {"train": 4656}, "virtual_train_size": 37248, "patience": 15}
        record = dict(records[-1], train_loss=0.1, val_wer=0.6, val_uwer=0.5, val_exact_rate=0.3, lr_used=0.001, lr=0.0005, seconds=10, vram_allocated_mib=3000, epoch_peak_vram_mib=4000)
        records[-1] = record
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            print_epoch_report(config, record, records, [{"file_name": "a.png", "reference": "a", "prediction": "b"}], 2)
        for label in ("VAL UWER", "Pente CER", "Gain fenetre", "Canaux", "CNN conv", "FILE: a.png", "REF : a", "PRED: b", "Best WER", "Sans gain CER"):
            self.assertIn(label, text.getvalue())

    def test_hub_publication_verifies_remote_hashes(self):
        from publish_hausa import publish_package
        with tempfile.TemporaryDirectory(prefix="hub-fixture-", dir=Path(__file__).parent) as folder:
            root = Path(folder)
            self.assertEqual(root.resolve().parent, Path(__file__).resolve().parent)
            (root / "best_cer.pt").write_bytes(b"synthetic fixture, not model weights")
            weight_hash = hashlib.sha256((root / "best_cer.pt").read_bytes()).hexdigest()
            manifest = {"author": "IX-World", "version": "v9", "files": {"best_cer.pt": weight_hash}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            blob = (root / "manifest.json").read_bytes()
            siblings = [SimpleNamespace(rfilename="best_cer.pt", lfs=SimpleNamespace(sha256=weight_hash)), SimpleNamespace(rfilename="manifest.json", lfs=None, blob_id=hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest())]
            with patch("publish_hausa.load_run"), patch("huggingface_hub.HfApi") as factory, contextlib.redirect_stdout(io.StringIO()):
                api = factory.return_value
                api.upload_folder.return_value = SimpleNamespace(oid="synthetic", commit_url="https://example.invalid/synthetic")
                api.model_info.return_value = SimpleNamespace(private=False, siblings=siblings)
                receipt = publish_package(root, "IX-World/synthetic-do-not-publish")
                self.assertEqual(receipt["verified_files"], 2)
                siblings[0].lfs.sha256 = "corrupt"
                with self.assertRaises(ValueError):
                    publish_package(root, "IX-World/synthetic-do-not-publish")

    def test_group_split_rejects_shared_sources_and_writer_links(self):
        rows = {part:[{"partition":"test" if part=="test" else "train", "file_name":f"{part}.png", "pixel_sha256":part, "source":source, "group_id":group}] for part,source,group in (("train","8","g8"),("validation","9","g9"),("test","1","g12"))}
        report = {"separation_unit":"source_and_known_scribe", "source_to_group":{"8":"g8","9":"g9","1":"g12","2":"g12"}}
        self.assertTrue(validate_partitions(rows,report,require_groups=True))
        bad = {part:[dict(row) for row in values] for part,values in rows.items()}
        bad["train"][0].update(source="2",group_id="g12")
        with self.assertRaises(ValueError):
            validate_partitions(bad,report,require_groups=True)
        with self.assertRaises(ValueError):
            validate_partitions(rows,{},require_groups=True)
        bad["train"][0].update(source="1",group_id="g12")
        with self.assertRaises(ValueError):
            validate_partitions(bad,report,require_groups=True)

    def test_validation_keeps_whole_groups_and_matches_target_without_scores(self):
        rows=[{"group_id":group,"file_name":f"{group}-{i}"} for group,n in (("a",15),("b",6),("c",4),("d",2)) for i in range(n)]
        train,val=group_validation(rows,0.1)
        self.assertEqual(len(train)+len(val),len(rows))
        self.assertEqual(len(val),2)
        self.assertFalse({r['group_id'] for r in train}&{r['group_id'] for r in val})

    def test_corrupt_release_archive_is_rejected_before_github_access(self):
        import publish_github_release
        with tempfile.TemporaryDirectory(prefix="archive-fixture-", dir=Path(__file__).parent) as folder:
            root = Path(folder)
            self.assertEqual(root.resolve().parent, Path(__file__).resolve().parent)
            archive = root / "corrupt.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("manifest.json", json.dumps({"author": "IX-World", "version": "v9", "files": {"best_cer.pt": "incorrect"}}))
                handle.writestr("best_cer.pt", b"corrupted fixture")
            with patch("publish_github_release.github_client") as connect:
                with self.assertRaises(ValueError):
                    publish_github_release.publish(SimpleNamespace(repo="IX-World/OCR-AJAMI-Hausa", asset=archive))
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
