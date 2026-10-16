"""Install command shortcuts alongside a local checkout; never starts training."""
import argparse
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
    for name in ["run_v9_v12.cmd", "test_v9_v12.cmd"] + [f"{action}_{version}.cmd" for action in ("lancer", "tester", "affiner") for version in ("v9", "v10", "v11", "v12")]:
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
