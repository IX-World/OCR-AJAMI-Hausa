import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from hausa_ocr import AUTHOR, SOURCE_REPOSITORY
from prepare_hausa import sha256, write_json


def load_run(run):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if config["author"] != AUTHOR or config["version"] not in ("v9", "v10", "v11", "v12"):
        raise ValueError("Version ou attribution non reconnue.")
    if metrics["examples"] != 400 or metrics["selection"] != "minimum_validation_CER":
        raise ValueError("La publication exige le resultat reel du test public de 400 exemples.")
    if sha256(run / "best_cer.pt") != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint different de celui evalue.")
    if not (metrics["split_sha256"] == config["split_sha256"] == protocol["split_sha256"] == sha256(run / "split.json")):
        raise ValueError("Split different entre configuration, checkpoint evalue et manifeste.")
    if sha256(run / "vocab.json") != config["vocab_sha256"]:
        raise ValueError("Vocabulaire modifie.")
    for name, expected in config["source_sha256"].items():
        if sha256(run / "source" / name) != expected:
            raise ValueError(f"Code source modifie : {name}")
    return config, metrics, protocol


def write_model_card(run):
    run = Path(run)
    config, metrics, protocol = load_run(run)
    version = config["version"].upper()
    counts = config["counts"]
    unknown = json.dumps(config["unseen_characters"], ensure_ascii=False)
    card = f'''---
language:
- ha
tags:
- ocr
- ajami
- handwritten-text-recognition
- crnn
- ctc
- IX-World
library_name: pytorch
datasets:
- IntelligenceResearchLab/Hausa
---

# Hausa Ajami OCR — {version} — contribution IX-World

## Attribution et portée

Code, adaptation expérimentale et entraînement : **IX-World**.
Dépôt du projet : {SOURCE_REPOSITORY}
Commit de code enregistré : `{config.get('source_commit')}`.
Le corpus provient d'IntelligenceResearchLab/Hausa ; sa provenance, ses crédits
et ses conditions de réutilisation restent ceux du dépôt d'origine.
Cette fiche ne réattribue pas les manuscrits ni les annotations à l'auteur du code.

Le modèle transforme une image de segment manuscrit haoussa Ajami en texte latin.
Il n'effectue pas de détection sur une page complète et n'exige aucune
transcription intermédiaire en caractères Ajami.

## Version des données et séparation des ensembles

- Révision du dataset : `{config['dataset_revision']}`.
- Entraînement : **{counts['train']} segments réels**.
- Validation : **{counts['validation']} segments**, tirés uniquement du train public.
- Test : les **400 exemples publics**, conservés dans leur partition d'origine.
- SHA-256 du split : `{config['split_sha256']}`.
- Graine du split : {protocol['seed']} ; stratification par identifiant de source.
- Les trois chemins invalides ont été corrigés ; la référence de « ranar »
  a été rétablie vers ajami_01289.png ; le doublon d'annotation identique a été supprimé.
- Les deux images identiques a_ajami_03400/03401, annotées « da »/« mutane »,
  sont exclues sur décision de l'utilisateur et conservées dans quarantine.json.
- Aucun doublon de pixels exact entre train, validation et test n'est présent.
  Les quasi-doublons et l'indépendance des copistes ne sont pas garantis.

Ce protocole diffère du split historique 2 942/96/160 : les scores de cette
campagne ne constituent pas une comparaison directe avec les anciens scores.
Tous les nouveaux V9 à V12 utilisent exactement le même split et vocabulaire.

## Architecture et prétraitement

{config['cnn_convolutions']} convolutions 3×3, BatchNorm et ReLU ;
deux premiers poolings 2×2, puis réduction verticale ; BiLSTM de deux couches,
256 unités par direction, dropout 0,2 ; projection linéaire et CTC.
Nombre de paramètres : **{config['parameter_count']:,}**.
Nombre de classes : **{config['classes']}**, BLANK et UNK compris.
Dropout convolutionnel : {config['cnn_dropout']}.

Hauteur 96 pixels, largeur proportionnelle, interpolation Lanczos,
miroir horizontal RTL ; niveaux de gris ramenés à [-1,1].
Nombre de canaux : {config['channels']}.
Pour V11/V12, second canal d'encre locale : fond flouté avec rayon 5 pixels,
I = clip(1,6 × max(fond − gris, 0), 0, 1), puis 2I−1.
Padding gris blanc (+1), et absence d'encre (−1) pour le second canal.
Le BiLSTM reçoit des séquences complétées, sans empaquetage.

Le vocabulaire est construit exclusivement sur le train ; les références
d'évaluation sont conservées en Unicode NFC, avec casse, espaces et diacritiques.
Caractères absents du train : `{unknown}`.
Une classe UNK représente les caractères inconnus ; aucune référence de test
n'est réécrite pour faciliter le calcul du CER/WER.

## Entraînement réellement configuré

- Initialisation aléatoire sans préentraînement ; graine {config['seed']}.
- Augmentation : `{config['augmentation']}` ; {config['virtual_train_size']} vues par époque.
- Les vues V10/V11/V12 sont fixes ; V9 applique l'augmentation dynamique à 30 % des segments.
- Batch {config['batch_size']}, groupes de largeur {config['bucket_size']}, workers 0.
- AdamW, LR initial 0,001, weight decay 0,0001, clipping de norme 5.
- ReduceLROnPlateau : CER validation, facteur 0,5, patience 3, minimum 0,000001.
- Maximum {config['max_epochs']} époques ; minimum {config['min_epochs']} avant arrêt ;
  patience {config['patience']}, amélioration minimale 0,0005 pour la patience.
- Sélection du checkpoint par minimum absolu du CER de validation.
- Float32 ; cuDNN benchmark ; déterminisme strict non imposé.
- GPU : {config['gpu']} ; PyTorch {config['torch']} ; CUDA {config['cuda']} ;
  Python {config['python']} ; Pillow {config['pillow']}.

Les adaptations communes de cette campagne sont décrites dans config.json,
notamment le vocabulaire train seul, le padding, le regroupement en lots
et l'arrêt anticipé harmonisé. Il ne s'agit pas d'une reproduction bit à bit
des anciens scripts à chemins C:\\ai-test.

## Résultats mesurés, sans résultat anticipé

| Mesure | Test public |
|---|---:|
| CER | {metrics['cer']*100:.4f} % |
| WER | {metrics['wer']*100:.4f} % |
| Segments exacts | {metrics['exact_rate']*100:.4f} % |
| Erreurs de caractères / caractères de référence | {metrics['character_errors']} / {metrics['reference_characters']} |
| Erreurs de mots / mots de référence | {metrics['word_errors']} / {metrics['reference_words']} |
| Checkpoint sélectionné à l'époque | {metrics['selected_epoch']} |
| Durée enregistrée d'entraînement | {metrics['elapsed_training_seconds']/60:.2f} min |
| Pic VRAM allouée | {metrics['peak_vram_mib']:.1f} MiB |

Le CER et le WER sont des rapports globaux des distances d'édition cumulées.
Les mots sont séparés par les espaces. Décodage glouton CTC, fusion des
répétitions et suppression de BLANK ; aucun lexique ni correcteur.
Les prédictions individuelles et les compteurs sont dans test_predictions.csv.
SHA-256 du checkpoint : `{metrics['checkpoint_sha256']}`.

## Reproduire et utiliser

Le code exact est archivé dans source/, avec ses empreintes dans config.json.
Installer PyTorch et Pillow, télécharger le dataset à la révision indiquée,
appliquer prepare_hausa.py avec le même split, puis :

```powershell
python source/train_hausa.py --version {config['version']} --dataset "CHEMIN_DATASET" --protocol "CHEMIN_PROTOCOLE" --output "CHEMIN_RUN" --test-only
```

Pour tester le package téléchargé, CHEMIN_RUN désigne son dossier racine.
Le modèle reste un checkpoint PyTorch personnalisé et n'est pas un modèle
Transformers chargé par AutoModel. La publication ne crée pas un serveur d'inférence.

## Limites et apport

Cette expérience documente une transcription directe Ajami–latin sous contraintes
de données et de mémoire GPU. L'association CNN–BiLSTM–CTC est établie ;
l'originalité éventuelle d'une adaptation doit être soutenue par des comparaisons
et ablations, pas par le seul nom de l'architecture ou l'utilisation d'une GPU de 6 Go.
Une seule graine ne mesure pas la variabilité entre entraînements. Des sources
et des références lexicales peuvent être partagées entre les partitions.
V13 reste une proposition : aucune amélioration ni aucun résultat V13 n'est revendiqué.
'''
    (run / "README.md").write_text(card, encoding="utf-8")


def package_run(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    config, metrics, _ = load_run(run)
    write_model_card(run)
    predictions = list(csv.DictReader((run / "test_predictions.csv").open(encoding="utf-8-sig", newline="")))
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    expected = {(row["partition"], row["file_name"]): row["transcript"] for row in split["test"]}
    if len(predictions) != 400 or len({(r["partition"], r["file_name"]) for r in predictions}) != 400:
        raise ValueError("Les predictions doivent couvrir exactement les 400 identifiants du test.")
    if any(expected.get((r["partition"], r["file_name"])) != r["reference"] for r in predictions):
        raise ValueError("References du test modifiees.")
    if sum(int(r["character_errors"]) for r in predictions) != metrics["character_errors"] or sum(int(r["word_errors"]) for r in predictions) != metrics["word_errors"]:
        raise ValueError("Compteurs des predictions differents des scores publies.")
    if destination.exists():
        raise FileExistsError(f"Le package existe deja : {destination}. Choisir un autre dossier.")
    destination.mkdir(parents=True)
    for name in ("best_cer.pt", "README.md", "config.json", "split.json", "vocab.json", "protocol.json", "quarantine.json", "history.csv", "test_metrics.json", "test_predictions.csv"):
        shutil.copy2(run / name, destination / name)
    shutil.copytree(run / "source", destination / "source")
    manifest = {str(path.relative_to(destination)).replace("\\", "/"): sha256(path) for path in sorted(destination.rglob("*")) if path.is_file()}
    write_json(destination / "manifest.json", {"author": AUTHOR, "version": config["version"], "files": manifest})
    archive = destination.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                handle.write(path, str(path.relative_to(destination)).replace("\\", "/"))
    print(f"Package verifie : {destination}\nArchive GitHub Release : {archive}", flush=True)


def publish_package(folder, repo):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["author"] != AUTHOR or manifest["version"] not in ("v9", "v10", "v11", "v12"):
        raise ValueError("Attribution ou version incorrecte.")
    for name, expected in manifest["files"].items():
        candidate = (folder / name).resolve()
        if not candidate.is_relative_to(folder.resolve()) or sha256(candidate) != expected:
            raise ValueError(f"Empreinte ou chemin incorrect : {name}")
    load_run(folder)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder), commit_message=f"IX-World: publish verified {manifest['version']} on Hausa public test 400")
    print(commit.commit_url, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IX-World : packager puis publier uniquement un modele entraine et evalue")
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    package.add_argument("--run", type=Path, required=True)
    package.add_argument("--destination", type=Path, required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--package", type=Path, required=True)
    publish.add_argument("--repo", required=True)
    args = parser.parse_args()
    if args.command == "package":
        package_run(args.run, args.destination)
    else:
        publish_package(args.package, args.repo)
