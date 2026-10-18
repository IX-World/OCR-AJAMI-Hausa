import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from publish_v13 import digest, finalize_package, validate_v13_run


class PublishV13Tests(unittest.TestCase):
    def fixture(self, root):
        context = {"defaults": dict(language="ha", locality=None, year_start=None, year_end=None, evidence="corpus"), "sources": {}}
        split = {part: [dict(partition="test" if part == "train" else "train", file_name=part + ".png", source=str(i), transcript="a")]
                 for i, part in enumerate(("train", "validation", "test"))}
        resolved = {str(i): context["defaults"] for i in range(3)}
        schema = dict(schema_version=1, languages=[None, "ha"], localities=[None], time_available=False, training_period_range=None)
        identity = dict(train_rows_sha256=digest(split["train"]), train_contexts={"0": context["defaults"]}, context_schema=schema,
                        annotations_sha256=digest({}), split_sha256="split", vocab_sha256="vocab", teacher_repo="teacher", teacher_revision="revision")
        manifest = dict(identity=identity, identity_sha256=digest(identity), cache_sha256="cache", training_only=True, teacher_frozen=True)
        config = dict(context_sha256=digest(context), resolved_context_sha256=digest(resolved), context_schema=schema,
                      semantic_identity_sha256=digest(identity), semantic_cache_sha256="cache", split_sha256="split", vocab_sha256="vocab",
                      teacher_repo="teacher", teacher_revision="revision", verified_contextual_annotations=0,
                      context_coverage={p: dict(examples=1, known_language=1, known_period=0, known_locality=0) for p in split})
        summary = dict(split_sha256="split", dataset_revision="dataset", selected_epoch=1, checkpoint_sha256="weights", partitions={})
        from hausa_ocr.metrics import UWER_DEFINITION
        for part, rows in split.items():
            row = rows[0]
            score = dict(examples=1, character_errors=0, reference_characters=1, word_errors=0, reference_words=1,
                         normalized_word_errors=0, normalized_reference_words=1, exact=1, cer=0, wer=0, uwer=0, exact_rate=1,
                         evaluated_partition=part, uwer_definition=UWER_DEFINITION,
                         **{k: v for k, v in summary.items() if k != "partitions"})
            summary["partitions"][part] = score
            prediction = dict(sample_id=row["partition"] + "/" + row["file_name"], file_name=row["file_name"], public_origin=row["partition"], source=row["source"],
                              reference="a", prediction="a", character_errors=0, word_errors=0, normalized_word_errors=0,
                              language="ha", locality="", year_start="", year_end="", evaluated_partition=part)
            with (root / f"{part}_predictions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=prediction)
                writer.writeheader()
                writer.writerow(prediction)
        with (root / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["partition", "cer", "wer", "uwer", "examples"])
            writer.writeheader()
            writer.writerows(dict(partition=p, **{f: values[f] for f in ("cer", "wer", "uwer", "examples")}) for p, values in summary["partitions"].items())
        for name, value in (("split.json", split), ("context.json", context), ("semantic_manifest.json", manifest), ("semantic_annotations.json", {})):
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        return config, summary

    @patch("publish_v13.read_epoch_history")
    def test_public_origin_is_distinct_from_experimental_role(self, history):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            config, summary = self.fixture(root)
            validate_v13_run(root, config, summary, {"group_disjoint": True})
            path = root / "train_predictions.csv"
            path.write_text(path.read_text().replace(",train\n", ",test\n"))
            with self.assertRaises(ValueError):
                validate_v13_run(root, config, summary, {"group_disjoint": True})

    @patch("publish_v13.read_epoch_history")
    def test_context_and_train_cache_provenance_cannot_be_changed(self, history):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            config, summary = self.fixture(root)
            path = root / "semantic_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["identity"]["train_rows_sha256"] = "validation_injected"
            manifest["identity_sha256"] = config["semantic_identity_sha256"] = digest(manifest["identity"])
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_v13_run(root, config, summary, {"group_disjoint": True})

    def test_nested_manifest_is_hashed_and_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder) / "package"
            (root / "semantics").mkdir(parents=True)
            (root / "semantics" / "manifest.json").write_text("{}")
            archive = finalize_package(root)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertIn("semantics/manifest.json", manifest["files"])
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(set(handle.namelist()), set(manifest["files"]) | {"manifest.json"})
            with self.assertRaises(FileExistsError):
                finalize_package(root)


if __name__ == "__main__":
    unittest.main()
