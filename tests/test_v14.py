import contextlib
import copy
import csv
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw

from hausa_ocr.data import LineDataset, collate
from hausa_ocr.models import build_model
from hausa_ocr.v14 import (TrialDataset, TensorBudgetSampler, collapse_tokens, minimum_word_risk,
                           train_group, training_candidates)
from train_v14 import DEFAULTS, better_selected, main as train_main, evaluate as real_evaluate, final_scores


class V14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, trial="A", **changes):
        return SimpleNamespace(trial=trial, **dict(DEFAULTS, **changes))

    def test_mwer_equals_exhaustive_ctc_risk_and_its_gradient(self):
        logits = torch.tensor([[0.3, 0.4, -0.2], [0.1, -0.2, 0.7], [0.4, 0.5, 0.1]], dtype=torch.double, requires_grad=True)
        lp = logits.log_softmax(-1)
        candidates = [(1,), (2,), ()]
        actual = minimum_word_risk(lp, candidates, "a", {"<BLANK>": 0, "a": 1, "b": 2})
        probabilities = []
        for candidate in candidates:
            probabilities.append(sum(lp.exp()[torch.arange(3), list(path)].prod() for path in itertools.product(range(3), repeat=3) if collapse_tokens(path) == candidate))
        q = torch.stack(probabilities)
        expected = (q[1] + q[2]) / q.sum()
        self.assertAlmostEqual(float(actual), float(expected), places=12)
        grad_actual, = torch.autograd.grad(actual, logits, retain_graph=True)
        grad_expected, = torch.autograd.grad(expected, logits)
        self.assertTrue(torch.allclose(grad_actual, grad_expected, atol=1e-10))

    def test_training_reference_is_always_in_candidates_without_duplicate(self):
        vocab = {"<BLANK>": 0, "<UNK>": 1, "a": 2, "b": 3}
        lp = torch.tensor([[3., -3., -3., 3.]] * 4).log_softmax(-1)
        candidates = training_candidates(lp, "aa", vocab, beam_width=2, nbest=1, token_top_k=1)
        self.assertIn((2, 2), candidates)
        self.assertEqual(len(candidates), len(set(candidates)))
        with self.assertRaises(ValueError):
            training_candidates(lp, "unknown", vocab)

    def test_fixed_preprocessing_is_v12_identical_and_dynamic_is_repeatable(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root / "data" / "train").mkdir(parents=True)
            image = Image.new("L", (96, 96), 230)
            ImageDraw.Draw(image).line((10, 20, 80, 70), fill=50, width=4)
            image.save(root / "data" / "train" / "a.png")
            row = dict(partition="train", split="train", file_name="a.png", transcript="da", source="8", width=96, height=96)
            vocab = {"<BLANK>": 0, "<UNK>": 1, "d": 2, "a": 3}
            old, fixed = LineDataset(root, [row], vocab, "v12", training=True), TrialDataset(root, [row], vocab, "fixed_8_views")
            for view in range(8):
                self.assertTrue(torch.equal(old[view][0], fixed[view][0]))
            clean = TrialDataset(root, [row], vocab)
            self.assertTrue(torch.equal(clean[0][0], LineDataset(root, [row], vocab, "v12")[0][0]))
            dynamic = TrialDataset(root, [row], vocab, "renewed_8_views")
            dynamic.epoch = 1
            first = dynamic[1][0]
            self.assertTrue(torch.equal(first, dynamic[1][0]))
            dynamic.epoch = 2
            self.assertFalse(torch.equal(first, dynamic[1][0]))
            with self.assertRaises(ValueError):
                TrialDataset(root, [dict(row, split="validation")], vocab, "renewed_8_views")

    def test_budget_includes_both_channels_and_preserves_every_view(self):
        class Fixture:
            expected_widths = [100, 500, 400, 100, 800, 90]
            def __len__(self):
                return len(self.expected_widths)
        sampler = TensorBudgetSampler(Fixture(), batch_size=4, max_elements=200000)
        batches = list(sampler)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(6)))
        self.assertTrue(all(len(b) * 192 * max(sampler.widths[i] for i in b) <= 200000 for b in batches))

    def group(self):
        group = []
        for n in (2, 1):
            group.append(dict(images=torch.randn(n, 2, 2, 8), widths=torch.full((n,), 8), targets=torch.tensor([2, 3] * n),
                              target_lengths=torch.full((n,), 2), rows=[dict(split="train", transcript="da") for _ in range(n)]))
        return group

    def test_sam_reuses_dropout_updates_batchnorm_once_and_restores_parameters(self):
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(2)
                self.dropout = torch.nn.Dropout(0.4)
                self.classifier = torch.nn.Linear(2, 4)
                self.masks = []
            def forward(self, images):
                x = self.dropout(self.bn(images))
                self.masks.append((x == 0).detach().clone())
                return self.classifier(x.mean(2).transpose(1, 2)[:, ::4])
        class RestoreCheckingSGD(torch.optim.SGD):
            def step(self, *args, **kwargs):
                for p, initial in zip(self.param_groups[0]["params"], self.originals):
                    if not torch.equal(p, initial):
                        raise AssertionError("SAM weights not restored before optimizer step")
                return super().step(*args, **kwargs)
        a = Small()
        c = copy.deepcopy(a)
        group = self.group()
        vocab = {"<BLANK>": 0, "<UNK>": 1, "d": 2, "a": 3}
        torch.manual_seed(99)
        train_group(a, group, torch.optim.SGD(a.parameters(), lr=0.01), vocab, self.args(), torch.device("cpu"), 1, 1)
        expected_rng = torch.get_rng_state()
        optimizer = RestoreCheckingSGD(c.parameters(), lr=0.01)
        optimizer.originals = [p.detach().clone() for p in c.parameters()]
        torch.manual_seed(99)
        train_group(c, group, optimizer, vocab, self.args("C"), torch.device("cpu"), 1, 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertEqual(int(c.bn.num_batches_tracked), 2)
        self.assertTrue(c.bn.track_running_stats)
        self.assertTrue(torch.equal(c.masks[0], c.masks[2]))
        self.assertTrue(torch.equal(c.masks[1], c.masks[3]))

    def test_uneven_accumulation_is_weighted_by_examples(self):
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(2, 4)
            def forward(self, images):
                return self.classifier(images.mean(2).transpose(1, 2)[:, ::4])
        a = Small()
        b = copy.deepcopy(a)
        group = self.group()
        opt_a, opt_b = torch.optim.SGD(a.parameters(), lr=0.01), torch.optim.SGD(b.parameters(), lr=0.01)
        vocab = {"<BLANK>": 0, "<UNK>": 1, "d": 2, "a": 3}
        train_group(a, group, opt_a, vocab, self.args(), torch.device("cpu"), 1, 1)
        logits = b(torch.cat([g["images"] for g in group]))
        criterion = torch.nn.CTCLoss(blank=0)
        criterion(logits.log_softmax(-1).transpose(0, 1), torch.cat([g["targets"] for g in group]), torch.tensor([2, 2, 2]), torch.tensor([2, 2, 2])).backward()
        torch.nn.utils.clip_grad_norm_(b.parameters(), 5.0)
        opt_b.step()
        self.assertTrue(all(torch.allclose(p, q, atol=1e-7) for p, q in zip(a.parameters(), b.parameters())))

    def test_sam_failure_restores_original_parameters(self):
        class Broken(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(2, 4)
                self.calls = 0
            def forward(self, images):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("synthetic second pass failure")
                return self.classifier(images.mean(2).transpose(1, 2)[:, ::4])
        model = Broken()
        old = [p.detach().clone() for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "synthetic second pass"):
            train_group(model, self.group()[:1], torch.optim.AdamW(model.parameters()),
                        {"<BLANK>": 0, "<UNK>": 1, "d": 2, "a": 3}, self.args("C"), torch.device("cpu"), 1, 1)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(model.parameters(), old)))

    def test_selection_never_accepts_worse_cer_to_improve_wer(self):
        old = dict(cer=0.4, wer=0.8)
        self.assertFalse(better_selected(dict(cer=0.41, wer=0.7), old, 0.4))
        self.assertTrue(better_selected(dict(cer=0.39, wer=0.7), old, 0.4))
        self.assertTrue(better_selected(dict(cer=0.39, wer=0.8), old, 0.4))

    def fixture(self, root):
        dataset, protocol = root / "dataset", root / "protocol"
        protocol.mkdir()
        splits = {}
        for i, part in enumerate(("train", "validation", "test"), 1):
            public = "test" if part != "train" else "train"
            path = dataset / "data" / public / f"{part}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("L", (32, 96), 245)
            ImageDraw.Draw(image).line((4, 8, 24, 80), fill=0, width=3)
            image.save(path)
            splits[part] = [dict(partition=public, split=part, file_name=path.name, transcript="da", source=str(i), width=32, height=96)]
        vocab = {"<BLANK>": 0, "<UNK>": 1, "d": 2, "a": 3}
        report = dict(split_sha256="synthetic", vocab_sha256="synthetic", counts={p: 1 for p in splits},
                      dataset_repo="synthetic", dataset_revision="synthetic", protocol_id="synthetic")
        from prepare_hausa import sha256
        for name, value in (("split.json", splits), ("vocab.json", vocab), ("protocol.json", report), ("quarantine.json", []), ("source_groups.json", {}), ("excluded_public_train.json", [])):
            (protocol / name).write_text(json.dumps(value), encoding="utf-8")
        report["split_sha256"], report["vocab_sha256"] = sha256(protocol / "split.json"), sha256(protocol / "vocab.json")
        (protocol / "protocol.json").write_text(json.dumps(report), encoding="utf-8")
        initial = root / "synthetic.pt"
        initial.write_bytes(b"synthetic fixture, not a published model")
        model = build_model("v12", len(vocab))
        checkpoint = dict(model_state_dict=model.state_dict(), config={"parameter_count": sum(p.numel() for p in model.parameters())})
        return dataset, protocol, splits, vocab, report, initial, checkpoint

    def test_all_trials_complete_without_test_then_explicit_final_test(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            dataset, protocol, splits, vocab, report, initial, checkpoint = self.fixture(root)
            for trial in "ABCD":
                output = root / trial
                argv = ["train_v14.py", "--trial", trial, "--dataset", str(dataset), "--protocol", str(protocol), "--initial-v12", str(initial), "--output", str(output),
                        "--device", "cpu", "--epochs", "1", "--min-epochs", "1", "--batch-size", "2", "--accumulate", "2", "--eval-batch-size", "2", "--show-examples", "0",
                        "--mwer-every", "1", "--mwer-beam-width", "3", "--mwer-nbest", "2", "--mwer-token-top-k", "0"]
                roles = []
                def tracked(model, batches, *args):
                    roles.append(batches.dataset.entries[0]["split"])
                    return real_evaluate(model, batches, *args)
                with patch("sys.argv", argv), patch("train_v14.load_protocol", return_value=(splits, vocab, report)), patch("train_v14.require_protocol"), \
                     patch("train_v14.load_v12", return_value=checkpoint), patch("train_v14.evaluate", side_effect=tracked), contextlib.redirect_stdout(io.StringIO()):
                    train_main()
                self.assertEqual(roles, ["validation", "validation", "train", "validation"])
                summary = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                self.assertFalse(summary["test_evaluated"])
                self.assertFalse((output / "test_metrics.json").exists())
                self.assertEqual(set(summary["partitions"]), {"train", "validation"})
                for name in ("selected.pt", "best_cer.pt", "best_wer.pt", "last.pt", "history.csv", "epochs.json", "EPOCHS.md", "README.md"):
                    self.assertTrue((output / name).is_file())
                with (output / "history.csv").open(newline="") as handle:
                    history = list(csv.DictReader(handle))
                self.assertEqual(len(history), 1)
                self.assertEqual(int(history[0]["mwer_examples"]) > 0, trial == "D")
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                scores = final_scores(build_model("v12", len(vocab)), config, splits, vocab, report,
                                      SimpleNamespace(trial="D", dataset=dataset, eval_batch_size=2), output, torch.device("cpu"), include_test=True)
            self.assertTrue(scores["test_evaluated"])
            self.assertEqual(set(scores["partitions"]), {"train", "validation", "test"})

    def test_interrupted_trial_resumes_optimizer_history_and_archived_sources(self):
        from hausa_ocr.v14 import train_epoch as real_train_epoch
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            dataset, protocol, splits, vocab, report, initial, checkpoint = self.fixture(root)
            output = root / "resume_C"
            argv = ["train_v14.py", "--trial", "C", "--dataset", str(dataset), "--protocol", str(protocol), "--initial-v12", str(initial), "--output", str(output),
                    "--device", "cpu", "--epochs", "2", "--min-epochs", "2", "--batch-size", "2", "--accumulate", "2", "--eval-batch-size", "2", "--show-examples", "0"]
            def interrupted(*args):
                if args[-2] == 2:
                    raise KeyboardInterrupt
                return real_train_epoch(*args)
            with patch("sys.argv", argv), patch("train_v14.load_protocol", return_value=(splits, vocab, report)), patch("train_v14.require_protocol"), \
                 patch("train_v14.load_v12", return_value=checkpoint), patch("train_v14.train_epoch", side_effect=interrupted), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    train_main()
            self.assertEqual(stopped.exception.code, 130)
            saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 1)
            first_steps = saved["global_step"]
            with patch("sys.argv", argv + ["--resume"]), patch("train_v14.__file__", str(output / "source" / "train_v14.py")), \
                 patch("train_v14.V12_CHECKPOINT_SHA256", saved["config"]["initial_checkpoint_sha256"]), \
                 patch("train_v14.load_protocol", return_value=(splits, vocab, report)), patch("train_v14.require_protocol"), contextlib.redirect_stdout(io.StringIO()):
                train_main()
            last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(last["epoch"], 2)
            self.assertGreater(last["global_step"], first_steps)
            with (output / "history.csv").open(newline="") as handle:
                self.assertEqual([int(r["epoch"]) for r in csv.DictReader(handle)], [1, 2])
            self.assertFalse(json.loads((output / "metrics.json").read_text())["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
