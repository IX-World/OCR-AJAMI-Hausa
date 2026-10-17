"""Install command shortcuts alongside a local checkout; never starts training."""
import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(r"D:\ai_test"))
    args = parser.parse_args()
    base = args.base.resolve()
    repo = Path(__file__).resolve().parent
    if repo.parent != base or repo.name != "OCR-AJAMI-Hausa":
        raise ValueError("Ce script doit etre execute dans BASE/OCR-AJAMI-Hausa.")
    context = base / "v13_context.json"
    if not context.exists():
        template = repo / "configs" / "v13_context.json"
        if not template.is_file():
            sparse = subprocess.run(["git", "-C", str(repo), "config", "--bool", "core.sparseCheckout"], capture_output=True, text=True)
            if sparse.returncode == 0 and sparse.stdout.strip() == "true":
                subprocess.run(["git", "-C", str(repo), "sparse-checkout", "add", "configs"], check=True)
        if not template.is_file():
            raise FileNotFoundError("Modele de contexte absent : recuperer configs/v13_context.json depuis Git avant l'installation.")
        shutil.copy2(template, context)
    for name in ["run_v9_v12.cmd", "test_v9_v12.cmd", "preparer_v13.cmd", "lancer_v13.cmd", "tester_v13.cmd", "reprendre_v13.cmd", "tester_beam_v10_validation.cmd"] + [f"{action}_{version}.cmd" for action in ("lancer", "tester", "affiner") for version in ("v9", "v10", "v11", "v12")]:
        text = f'@echo off\r\ncall "%~dp0OCR-AJAMI-Hausa\\{name}" %*\r\nexit /b %errorlevel%\r\n'
        target = base / name
        if target.exists() and target.read_bytes() != text.encode("ascii"):
            raise FileExistsError(f"Un autre raccourci existe : {target}")
        target.write_bytes(text.encode("ascii"))
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "IX-World"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "116719527+IX-World@users.noreply.github.com"], check=True)
    print(f"Commandes individuelles et notebooks prets dans {base}")


if __name__ == "__main__":
    main()
