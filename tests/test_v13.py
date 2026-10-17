import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from hausa_ocr.semantic import (context_values, fit_context_schema, load_annotations, load_context,
                                semantic_loss, text_input, validate_context)
from hausa_ocr.v13 import PixelBudgetSampler, SemanticCRNN
from prepare_v13_semantics import encode_targets
from train_v13 import main as train_main, evaluate as real_evaluate


def context(language="ha", locality=None, start=None, end=None):
    return dict(language=language, locality=locality, year_start=start, year_end=end, evidence="Verified synthetic fixture")


class ContextTests(unittest.TestCase):
    def test_schema_fitted_only_on_train_and_unknown_values_masked(self):
        rows = [{"source": "1"}]
        contexts = {"1": context(locality="region_train"), "2": context(locality="held_out_region", start=1800, end=1850), "3": context(None)}
        schema = fit_context_schema(rows, contexts)
        self.assertEqual(schema["localities"], [None, "region_train"])
        encoded = context_values([{"source": "2"}, {"source": "3"}], contexts, schema)
        self.assertEqual(encoded["locality"].tolist(), [0, 0])
        self.assertEqual(encoded["language"].tolist(), [1, 0])
        self.assertEqual(encoded["time"][0, 0], 0)  # No dated train example: no untrained temporal branch at evaluation.
        self.assertTrue(torch.equal(encoded["time"][1], torch.zeros(3)))

    def test_unverified_context_and_invalid_or_unsupported_metadata_rejected(self):
        for ctx in [context(start=1800), context(start=1900, end=1800), context(language="ar"), dict(context(), evidence="")]:
            with self.assertRaises(ValueError):
                validate_context(ctx)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "context.json"
            p.write_text(json.dumps(dict(schema_version=1, calendar="CE", defaults=context(), sources={"1": {"locality": "imagined"}})), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_context(p, {"1"})

    def test_annotations_admit_only_train_ids_even_with_raw_public_test_origin(self):
        rows = [{"partition": "test", "file_name": "reassigned_train.png", "transcript": "da"}]
        annotation = dict(sample_id="test/reassigned_train.png", verified=True, evidence="Training source only", context_before="before", context_after="after")
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "annotations.jsonl"
            p.write_text(json.dumps(annotation), encoding="utf-8")
            loaded = load_annotations(p, rows)
            self.assertEqual(text_input(rows[0], loaded[annotation["sample_id"]]), ("before da after", 7, 9))
            for record in [dict(annotation, sample_id="test/held_out.png"), dict(annotation, verified=False)]:
                p.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_annotations(p, rows)


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_multi_positive_duplicates_and_missing_language(self):
        vector = torch.randn(1, 768)
        visual = vector.repeat(3, 1).requires_grad_()
        total, cosine, contrastive = semantic_loss(visual, vector.repeat(3, 1), ["da"] * 3, torch.ones(3, dtype=torch.long))
        self.assertLess(abs(float(total)), 1e-5)
        self.assertEqual(float(contrastive), 0.0)
        unknown, _, _ = semantic_loss(visual, vector.repeat(3, 1), ["da"] * 3, torch.zeros(3, dtype=torch.long))
        unknown.backward()
        self.assertTrue(torch.equal(visual.grad, torch.zeros_like(visual)))

    def test_auxiliary_semantic_gradient_reaches_image_encoder(self):
        schema = {"languages": [None, "ha"], "localities": [None, "region"]}
        model = SemanticCRNN(5, schema)
        images = torch.randn(2, 1, 96, 64)
        ctx = {"language": torch.tensor([1, 1]), "locality": torch.tensor([1, 1]), "time": torch.tensor([[1., 0.1, 0.3], [1., 0.1, 0.3]])}
        logits, visual = model(images, torch.tensor([64, 48]), ctx, True)
        loss, _, _ = semantic_loss(visual, torch.randn(2, 768), ["da", "mutane"], ctx["language"])
        loss.backward()
        self.assertEqual(logits.shape, (2, 16, 5))
        for parameter in [model.cnn[0].weight, model.semantic[0].weight, model.time[0].weight, model.locality.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_same_spelling_with_different_contexts_is_not_a_forced_negative(self):
        visual = torch.randn(2, 768, requires_grad=True)
        target = torch.randn(2, 768)
        loss, cosine, contrastive = semantic_loss(visual, target, ["same", "same"], torch.ones(2, dtype=torch.long), positive_keys=["context_1800", "context_2000"])
        self.assertEqual(float(contrastive), 0.0)
        self.assertGreater(float(cosine), 0)
        loss.backward()
        self.assertGreater(float(visual.grad.abs().sum()), 0)

    def test_budget_sampler_preserves_every_example(self):
        class Fixture:
            expected_widths = [100, 400, 200, 700, 50]
            def __len__(self):
                return len(self.expected_widths)
        for training in [False, True]:
            sampler = PixelBudgetSampler(Fixture(), batch_size=3, max_pixels=100000, training=training)
            batches = list(sampler)
            self.assertEqual(sorted(i for b in batches for i in b), list(range(5)))
            self.assertTrue(all(len(b) <= 3 and len(b) * max(sampler.widths[i] for i in b) * 96 <= 100000 for b in batches))

    def test_teacher_pooling_uses_reference_span_without_context_or_specials(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": torch.tensor([[0, 7, 8, 9, 2]]), "attention_mask": torch.ones(1, 5, dtype=torch.long),
                        "offset_mapping": torch.tensor([[[0, 0], [0, 3], [4, 6], [7, 10], [0, 0]]]),
                        "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 1]])}
        class Teacher:
            def __call__(self, **kwargs):
                return SimpleNamespace(last_hidden_state=torch.tensor([[[100., 100.], [0., 1.], [3., 0.], [0., 1.], [100., 100.]]]))
        result = encode_targets(Teacher(), Tokenizer(), [dict(text="xxx da yyy", start=4, end=6)], 1)
        self.assertTrue(torch.equal(result, torch.tensor([[1., 0.]])))

    def test_synthetic_full_epoch_final_metrics_and_reload(self):
        # Deliberately synthetic, mocked protocol: never a published OCR result.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset, protocol, output = root / "dataset", root / "protocol", root / "run"
            protocol.mkdir()
            semantics = root / "semantics"
            semantics.mkdir()
            splits = {}
            for index, part in enumerate(["train", "validation", "test"], 1):
                public = "train" if part != "test" else "test"
                path = dataset / "data" / public / f"{part}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (64, 96), 255).save(path)
                splits[part] = [dict(partition=public, file_name=path.name, transcript="da", source=str(index), width=64, height=96)]
            for name in ["split.json", "vocab.json", "protocol.json", "quarantine.json", "source_groups.json", "excluded_public_train.json"]:
                (protocol / name).write_text("{}", encoding="utf-8")
            for name in ["manifest.json", "annotations.json"]:
                (semantics / name).write_text("{}", encoding="utf-8")
            context_path = root / "context.json"
            context_path.write_text(json.dumps(dict(schema_version=1, calendar="CE", defaults=context(), sources={})), encoding="utf-8")
            report = dict(split_sha256="synthetic", vocab_sha256="synthetic", counts={p: 1 for p in splits}, dataset_repo="synthetic", dataset_revision="synthetic",
                          protocol_id="synthetic", source_to_group={str(i): str(i) for i in [1, 2, 3]})
            vocab = {"<BLANK>": 0, "<UNK>": 1, "d": 2, "a": 3}
            cache = dict(targets=torch.randn(1, 768), context_schema={"schema_version": 1, "languages": [None, "ha"], "localities": [None]},
                         identity=dict(teacher_repo="synthetic", teacher_revision="synthetic"), identity_sha256="synthetic", verified_contextual_annotations=0, positive_keys=["synthetic"])
            argv = ["train_v13.py", "--dataset", str(dataset), "--protocol", str(protocol), "--context", str(context_path), "--semantics", str(semantics),
                    "--output", str(output), "--device", "cpu", "--epochs", "1", "--min-epochs", "1", "--batch-size", "2", "--accumulate", "2", "--show-examples", "0"]
            evaluation_roles = []
            def inspect_evaluation(model, batches, *args):
                evaluation_roles.append(batches.dataset.entries[0]["file_name"].removesuffix(".png"))
                return real_evaluate(model, batches, *args)
            with patch("sys.argv", argv), patch("train_v13.load_protocol", return_value=(splits, vocab, report)), patch("train_v13.require_frozen_split"), \
                 patch("train_v13.evaluate", side_effect=inspect_evaluation), \
                 patch("train_v13.load_semantics", return_value=(cache, {"train/train.png": 0}, "synthetic")), contextlib.redirect_stdout(io.StringIO()):
                train_main()
                with patch("sys.argv", argv + ["--evaluate-only"]):
                    train_main()
            self.assertEqual(evaluation_roles, ["validation", "train", "validation", "test", "train", "validation", "test"])
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(metrics["partitions"]), {"train", "validation", "test"})
            for part, values in metrics["partitions"].items():
                self.assertEqual(values["examples"], 1)
                self.assertEqual(values["evaluated_partition"], part)
                self.assertIn("uwer", values)
                self.assertFalse(values["augmented"])
            with (output / "history.csv").open(encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            self.assertEqual(len(history), 1)
            self.assertIn("train_semantic_loss", history[0])
            self.assertTrue((output / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
