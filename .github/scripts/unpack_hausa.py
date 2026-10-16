import sys
import zipfile
from pathlib import Path

archives = list(Path(sys.argv[1]).glob("*.zip"))
if len(archives) != 1:
    raise SystemExit("Une archive ZIP exactement est attendue.")
target = Path(sys.argv[2]).resolve()
target.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(archives[0]) as archive:
    for member in archive.infolist():
        path = (target / member.filename).resolve()
        if not path.is_relative_to(target) or member.file_size > 2_000_000_000:
            raise SystemExit("Chemin ou taille invalide dans l'archive.")
    if sum(member.file_size for member in archive.infolist()) > 4_000_000_000:
        raise SystemExit("Archive trop volumineuse.")
    archive.extractall(target)
