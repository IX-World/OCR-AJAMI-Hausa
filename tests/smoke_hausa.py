"""Short CUDA checks; synthetic optimizer steps never create trained-model results."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_hausa
from hausa_ocr.models import build_model

root = Path(__file__).resolve().parents[1]
base = Path(r"D:\ai_test")
for version in ("v9", "v10", "v11", "v12"):
    with patch.object(sys, "argv", ["train_hausa.py", "--version", version, "--dataset", str(base / "Hausa_repo_2026-10-15"), "--protocol", str(base / "protocol_hausa_20261015"), "--check-only"]):
        train_hausa.main()

for version in ("v9", "v12"):
    channels = 1 if version == "v9" else 2
    model = build_model(version, 12).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(2, channels, 96, 64, device="cuda")
    logits = model(images)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(
        logits.log_softmax(-1).permute(1, 0, 2),
        torch.tensor([2, 2, 3, 4], device="cuda"),
        torch.tensor([16, 16]), torch.tensor([2, 2]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    archive = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, archive)
    archive.seek(0)
    saved = torch.load(archive, map_location="cpu", weights_only=True)
    restored = build_model(version, 12).cuda()
    restored.load_state_dict(saved["model"])
    model.eval()
    restored.eval()
    with torch.inference_mode():
        assert torch.allclose(model(images), restored(images), rtol=1e-4, atol=1e-5)
    print(f"{version}: synthetic CTC backward and checkpoint round trip OK", flush=True)
    del model, restored, optimizer, logits, loss, saved, archive
    torch.cuda.empty_cache()

for path in (root / "notebooks").glob("*.ipynb"):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec")
        assert not cell.get("outputs"), "Notebook contains fabricated/pre-run results"
    print(f"Notebook syntax and empty outputs OK: {path.name}", flush=True)
