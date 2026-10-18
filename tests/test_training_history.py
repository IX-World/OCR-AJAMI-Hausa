import csv
import json
import tempfile
import unittest
import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hausa_ocr.publication_history import UNITS, V13_UNITS, read_epoch_history, write_epoch_report


class TrainingHistoryTests(unittest.TestCase):
    def fixture(self, root):
        records = []
        for epoch, cer in ((1, 0.8), (2, 0.5), (3, 0.6)):
            row = dict.fromkeys(UNITS, 0.0)
            row.update(epoch=epoch, train_loss=1 / epoch, val_cer=cer, val_wer=0.9,
                       val_uwer=0.85, val_exact_rate=0.1, lr_used=0.001,
                       lr=0.0005 if epoch == 3 else 0.001, seconds=1.2,
                       val_cer_slope="" if epoch == 1 else -0.1,
                       val_cer_window_gain="" if epoch == 1 else 0.2)
            records.append(row)
        with (root / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNITS)
            writer.writeheader()
            writer.writerows(records)
        selected = dict(selected_epoch=2, checkpoint_sha256="weights", split_sha256="split",
                        validation_at_selected_epoch=dict(cer=0.5, wer=0.9, uwer=0.85, exact_rate=0.1))
        (root / "test_metrics.json").write_text(json.dumps(selected), encoding="utf-8")

    def test_export_preserves_values_units_and_missing_measurements(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            self.fixture(root)
            original = (root / "history.csv").read_bytes()
            report = write_epoch_report(root)
            self.assertEqual((root / "history.csv").read_bytes(), original)
            self.assertEqual(report["completed_epochs"], 3)
            self.assertEqual(report["epochs"][1]["val_cer"], 0.5)
            self.assertIsNone(report["epochs"][0]["val_cer_slope"])
            self.assertNotIn("test_cer", report["epochs"][0])
            self.assertNotIn("train_cer", report["epochs"][0])
            text = (root / "EPOCHS.md").read_text(encoding="utf-8")
            self.assertIn("**2 retenue**", text)
            self.assertIn("50.0000", text)
            self.assertIn("ne sont pas enregistrés à chaque époque", text)

    def test_wrong_selected_epoch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            self.fixture(root)
            selected = json.loads((root / "test_metrics.json").read_text())
            selected["selected_epoch"] = 3
            (root / "test_metrics.json").write_text(json.dumps(selected))
            with self.assertRaises(ValueError):
                read_epoch_history(root)

    def v13_fixture(self, root):
        self.fixture(root)
        selected = json.loads((root / "test_metrics.json").read_text())
        selected["version"] = "v13"
        validation = dict(selected.pop("validation_at_selected_epoch"), selected_epoch=2, checkpoint_sha256="weights")
        (root / "validation_metrics.json").write_text(json.dumps(validation))
        (root / "test_metrics.json").write_text(json.dumps(selected))
        with (root / "history.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.update(train_ctc_loss=float(row["train_loss"]) - 0.1, train_semantic_loss=1.0,
                       train_cosine_loss=0.5, train_contrastive_loss=1.5, semantic_weight=0.1, optimizer_steps=4)
        with (root / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=V13_UNITS)
            writer.writeheader()
            writer.writerows(rows)

    def test_v13_combined_loss_and_auxiliary_units_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            self.v13_fixture(root)
            original = (root / "history.csv").read_bytes()
            report = write_epoch_report(root)
            self.assertEqual((root / "history.csv").read_bytes(), original)
            self.assertIn("CTC + semantic_weight", report["units"]["train_loss"])
            self.assertEqual(report["epochs"][0]["optimizer_steps"], 4)
            self.assertIn("Décomposition des pertes V13", (root / "EPOCHS.md").read_text(encoding="utf-8"))
            self.assertNotIn("moyenne des pertes CTC des batches", (root / "EPOCHS.md").read_text(encoding="utf-8"))

    def test_v13_inconsistent_combined_loss_and_selected_validation_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            self.v13_fixture(root)
            with (root / "history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["train_ctc_loss"] = 0.7
            with (root / "history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=V13_UNITS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                read_epoch_history(root)
            self.v13_fixture(root)
            validation = json.loads((root / "validation_metrics.json").read_text())
            validation["wer"] = 0.1
            (root / "validation_metrics.json").write_text(json.dumps(validation))
            with self.assertRaises(ValueError):
                read_epoch_history(root)

    def test_missing_epoch_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            self.fixture(root)
            data = (root / "history.csv").read_text()
            (root / "history.csv").write_text(data.replace("\n3,", "\n4,", 1))
            with self.assertRaises(ValueError):
                read_epoch_history(root)

            self.fixture(root)
            data = (root / "history.csv").read_text()
            (root / "history.csv").write_text(data.replace("0.8,", "nan,", 1))
            with self.assertRaises(ValueError):
                read_epoch_history(root)

    def test_modified_extra_release_asset_is_rejected_before_network(self):
        from publish_github_release import publish
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            split = {part: [{"partition": "train", "file_name": part + ".png",
                            "pixel_sha256": part, "source": source, "group_id": source, "split": part}]
                     for part, source in (("train", "8"), ("validation", "15"), ("test", "3"))}
            protocol = {"separation_unit": "source_and_known_scribe", "public_test_unchanged": False,
                        "all_usable_images_used": True, "source_to_group": {"8": "8", "15": "15", "3": "3"}}
            files = {"protocol.json": json.dumps(protocol).encode(), "split.json": json.dumps(split).encode(), "history.csv": b"authentic"}
            manifest = {"author": "IX-World", "version": "v9", "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
            archive = root / "model.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for name, data in files.items():
                    handle.writestr(name, data)
                handle.writestr("manifest.json", json.dumps(manifest))
            extra = root / "history.csv"
            extra.write_bytes(b"modified")
            with patch("publish_github_release.github_client") as connect:
                with self.assertRaises(ValueError):
                    publish(SimpleNamespace(repo="IX-World/OCR-AJAMI-Hausa", asset=archive, extra_asset=[extra]))
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
