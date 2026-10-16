import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from hausa_ocr import AUTHOR, SOURCE_REPOSITORY
from hausa_ocr.metrics import UWER_DEFINITION
from hausa_ocr.protocol import validate_partitions
from hausa_ocr.training_history import write_epoch_report
from prepare_hausa import sha256, write_json


def load_run(run, require_group_disjoint=False):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if config.get("protocol_sha256") and sha256(run/"protocol.json") != config["protocol_sha256"]:
        raise ValueError("Le protocole archive a ete modifie depuis l'entrainement.")
    splits = json.loads((run / "split.json").read_text(encoding="utf-8"))
    validate_partitions(splits, protocol, require_groups=require_group_disjoint)
    if require_group_disjoint and not protocol.get("all_usable_images_used",False):
        raise ValueError("La production actuelle exige le protocole utilisant tout le corpus valide.")
    if protocol.get("separation_unit") == "source_and_known_scribe":
        for name,field in (("source_groups.json","source_groups_sha256"),("excluded_public_train.json","excluded_public_train_sha256")):
            if sha256(run/name) != protocol[field]:
                raise ValueError("Manifeste de separation modifie dans le run.")
    if config["author"] != AUTHOR or config["version"] not in ("v9", "v10", "v11", "v12"):
        raise ValueError("Version ou attribution non reconnue.")
    if metrics["examples"] != config["counts"]["test"] or metrics["selection"] != "minimum_validation_CER":
        raise ValueError("La publication exige le resultat reel du test fige dans le protocole.")
    if sha256(run / "best_cer.pt") != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint different de celui evalue.")
    if not (metrics["split_sha256"] == config["split_sha256"] == protocol["split_sha256"] == sha256(run / "split.json")):
        raise ValueError("Split different entre configuration, checkpoint evalue et manifeste.")
    if sha256(run / "vocab.json") != config["vocab_sha256"]:
        raise ValueError("Vocabulaire modifie.")
    for name, expected in config["source_sha256"].items():
        if sha256(run / "source" / name) != expected:
            raise ValueError(f"Code source modifie : {name}")
    load_complete_metrics(run, config, metrics)
    load_split_audit(run, config, metrics)
    return config, metrics, protocol


def load_complete_metrics(run, config, test_metrics):
    path = run / "metrics.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["checkpoint_sha256"] != test_metrics["checkpoint_sha256"] or summary["split_sha256"] != config["split_sha256"]:
        raise ValueError("Le bilan complet ne correspond pas au checkpoint ou au split evalue.")
    if summary["uwer_definition"] != UWER_DEFINITION or set(summary["partitions"]) != {"train", "validation", "test"}:
        raise ValueError("Definition UWER ou partitions du bilan incorrectes.")
    for part, values in summary["partitions"].items():
        if values["examples"] != config["counts"][part] or values["checkpoint_sha256"] != summary["checkpoint_sha256"] or values["augmented"]:
            raise ValueError(f"Evaluation incoherente pour {part}.")
        if json.loads((run / f"{part}_metrics.json").read_text(encoding="utf-8")) != values:
            raise ValueError(f"Scores {part} differents du bilan complet.")
    for name, expected in summary["evaluation_source_sha256"].items():
        if sha256(run / "evaluation_source" / name) != expected:
            raise ValueError(f"Code d'evaluation modifie : {name}")
    return summary


def load_split_audit(run, config, metrics):
    names = ("split_audit.json", "split_audit.md", "audit_hausa_run.py")
    if not any((run / name).exists() for name in names):
        return None
    if not all((run / name).is_file() for name in names):
        raise ValueError("Audit de split incomplet.")
    audit = json.loads((run / "split_audit.json").read_text(encoding="utf-8"))
    if audit["checkpoint_sha256"] != metrics["checkpoint_sha256"] or audit["split_sha256"] != config["split_sha256"]:
        raise ValueError("Audit d'un autre checkpoint ou split.")
    if audit["audit_code_sha256"] != sha256(run / "audit_hausa_run.py") or audit["markdown_sha256"] != sha256(run / "split_audit.md"):
        raise ValueError("Code ou compte rendu de l'audit modifie.")
    expected_public_test=json.loads((run/"protocol.json").read_text(encoding="utf-8")).get("public_test_unchanged",True)
    if audit["counts"] != config["counts"] or audit["public_test_preserved"] != expected_public_test or not audit["public_train_coverage_complete"]:
        raise ValueError("Couverture du corpus incoherente dans l'audit.")
    if any(any(values.values()) for values in audit["cross_partition_overlap"].values()):
        raise ValueError("Chevauchement dans l'audit de split.")
    return audit


def write_model_card(run):
    run = Path(run)
    config, metrics, protocol = load_run(run)
    version = config["version"].upper()
    counts = config["counts"]
    unknown = json.dumps(config["unseen_characters"], ensure_ascii=False)
    initialization = "Initialisation aléatoire sans préentraînement" if config["from_scratch"] else f"Affinage depuis le checkpoint {config['initial_checkpoint_sha256']} sélectionné à l'époque {config['initial_checkpoint_epoch']}, avec un nouvel optimiseur"
    strict = protocol.get("separation_unit") == "source_and_known_scribe"
    full = protocol.get("all_usable_images_used",False)
    separation = f"Séparation par manuscrit et copiste identifié. {protocol['excluded_public_train_count']} images du train public liées aux groupes du test sont exclues. Les sources 1 et 2 sont regroupées. Les identités de copistes non documentées restent une limite. Voir source_groups.json et excluded_public_train.json." if strict else "Protocole initial séparé par images : des manuscrits sont partagés. Ces résultats sont historiques et ne satisfont pas la nouvelle campagne séparée par provenance."
    validation_rule = "groupes de provenance entiers, avec effectif le plus proche de la cible 10 %" if strict else "stratification des images par source"
    preparation = "Appliquer prepare_hausa.py pour les corrections initiales, puis prepare_hausa_grouped.py pour la séparation par provenance. Les fichiers source_groups.json et excluded_public_train.json décrivent le nouveau protocole." if strict else "Appliquer prepare_hausa.py avec le split initial archivé."
    test_rule = f"Test : **{counts['test']} exemples**, sélectionnés par groupes sur le corpus complet ; le test public initial n'est pas conservé comme partition." if full else f"Test : les **{counts['test']} exemples publics**, conservés dans leur partition d'origine."
    if full:
        separation = "Tous les exemples valides du corpus HF sont répartis entre les trois ensembles, sans manuscrit partagé. Les sources 1 et 2 sont regroupées par copiste connu. Seules les deux annotations contradictoires restent en quarantaine. L'identité des autres copistes n'est pas documentée."
        validation_rule = "groupes entiers au plus près de 200 exemples de validation, puis de 400 exemples de test ; toutes les autres images sont entraînées"
        preparation += " Utiliser impérativement --all-images pour retrouver ce split."
    source_limit = "Aucune source n'est partagée entre les partitions ; les références lexicales peuvent se répéter. L'indépendance des copistes non identifiés et l'absence de quasi-doublons restent à vérifier." if strict else "Des sources et des références lexicales peuvent être partagées entre les partitions."
    complete = load_complete_metrics(run, config, metrics)
    measured_limit = ""
    if complete:
        train_scores = complete["partitions"]["train"]
        measured_limit = f"Le WER du train ({train_scores['wer']:.2%}) et celui du test ({metrics['wer']:.2%}) montrent un écart de généralisation. Le WER de test est un taux d'erreur, pas un taux de réussite. Ce checkpoint est publié comme résultat expérimental reproductible ; ces scores ne démontrent pas une transcription autonome fiable."
    with (run / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        completed_epochs = len(list(csv.DictReader(handle)))
    write_epoch_report(run)
    epoch_curves = "![Courbes des mesures enregistrées pendant les époques](training_curves.png)\n\n[Version vectorielle SVG](training_curves.svg)." if (run / "training_curves.png").is_file() else ""
    audit_section = (run / "split_audit.md").read_text(encoding="utf-8") if load_split_audit(run, config, metrics) else ""
    if complete:
        result_table = "| Partition | Segments | CER | WER | UWER | Segments exacts |\n|---|---:|---:|---:|---:|---:|\n"
        for part, values in complete["partitions"].items():
            result_table += f"| {part} | {values['examples']} | {values['cer']:.4%} | {values['wer']:.4%} | {values['uwer']:.4%} | {values['exact_rate']:.4%} |\n"
        result_table += "\nLes trois partitions sont évaluées avec **le même checkpoint**, sélectionné\nsur le CER de validation. Le train est réévalué sur chaque image originale\nune seule fois, sans augmentation ; son score mesure l'ajustement aux données vues.\nLes scores de validation interviennent dans la sélection ; seul le test reste tenu à l'écart.\n\n"
        result_table += "Les compteurs bruts et normalisés, les dénominateurs et les empreintes sont\ndans metrics.json et dans les trois fichiers *_metrics.json. Le tableau est\négalement exporté dans metrics.csv ; les prédictions sont dans train_predictions.csv,\nvalidation_predictions.csv et test_predictions.csv. Le code d'évaluation est\narchivé séparément dans evaluation_source/, avec ses empreintes dans metrics.json.\n"
        result_table += f"\nDéfinition de l'UWER dans ce projet : {UWER_DEFINITION}\nL'UWER ne remplace pas le WER strict ; la ponctuation reste évaluée.\n"
    else:
        result_table = f"| Mesure | Test du protocole |\n|---|---:|\n| CER | {metrics['cer']:.4%} |\n| WER | {metrics['wer']:.4%} |\n| Segments exacts | {metrics['exact_rate']:.4%} |\n\nLe bilan train/validation/test et l'UWER ne sont pas encore calculés pour ce run.\nExécuter le script actuel avec --evaluate-only une fois l'entraînement terminé.\n"
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
Phase : `{config.get('experiment_stage', 'from_scratch')}`.
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
- Validation : **{counts['validation']} segments**, sélectionnés selon la règle ci-dessous.
- {test_rule}
- SHA-256 du split : `{config['split_sha256']}`.
- Construction de la validation : {validation_rule} ; graine de référence {protocol['seed']}.
- Les trois chemins invalides ont été corrigés ; la référence de « ranar »
  a été rétablie vers ajami_01289.png ; le doublon d'annotation identique a été supprimé.
- Les deux images identiques a_ajami_03400/03401, annotées « da »/« mutane »,
  sont exclues sur décision de l'utilisateur et conservées dans quarantine.json.
- Aucun doublon de pixels exact entre train, validation et test n'est présent.
  Les quasi-doublons et l'indépendance des copistes ne sont pas garantis.

Ce protocole diffère du split historique 2 942/96/160 : les scores de cette
campagne ne constituent pas une comparaison directe avec les anciens scores.
Tous les nouveaux V9 à V12 utilisent exactement le même split et vocabulaire.

{separation}

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

- {initialization} ; graine {config['seed']}.
- Augmentation : `{config['augmentation']}` ; {config['virtual_train_size']} vues par époque.
- Les vues V10/V11/V12 sont fixes ; V9 applique l'augmentation dynamique à 30 % des segments.
- Batch {config['batch_size']}, groupes de largeur {config['bucket_size']}, workers 0.
- AdamW, LR initial {config['learning_rate']:g}, weight decay 0,0001, clipping de norme 5.
- ReduceLROnPlateau : CER validation, facteur 0,5, patience 3, minimum 0,000001.
- Maximum {config['max_epochs']} époques ; minimum {config['min_epochs']} avant arrêt ;
  patience {config['patience']}, amélioration minimale 0,0005 pour la patience.
- Sélection du checkpoint par minimum absolu du CER de validation.
- Époques effectivement exécutées : {completed_epochs} ; checkpoint publié : époque {metrics['selected_epoch']}.
- Float32 ; cuDNN benchmark ; déterminisme strict non imposé.
- GPU : {config['gpu']} ; PyTorch {config['torch']} ; CUDA {config['cuda']} ;
  Python {config['python']} ; Pillow {config['pillow']}.

Les adaptations communes de cette campagne sont décrites dans config.json,
notamment le vocabulaire train seul, le padding, le regroupement en lots
et l'arrêt anticipé harmonisé. Il ne s'agit pas d'une reproduction bit à bit
des anciens scripts à chemins C:\\ai-test.

## Résultats mesurés, sans résultat anticipé

{result_table}

## Historique de toutes les époques

Les **{completed_epochs} époques enregistrées** sont disponibles dans
[le tableau complet avec les unités](EPOCHS.md), le [CSV brut](history.csv)
et le [JSON numérique](epochs.json). Le tableau indique les meilleurs scores
de validation, le checkpoint retenu, la loss train, les CER/WER/UWER et le taux
exact de validation, le LR utilisé et suivant, le temps, la VRAM, la pente
et le gain du CER. Les courbes sont incluses lorsqu'elles ont été générées.
Les CER/WER/UWER train et test ne sont pas mesurés à chaque époque : leur bilan
final utilise le checkpoint retenu sur la validation.

{epoch_curves}

| Provenance et ressources | Valeur |
|---|---:|
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

{audit_section}

## Reproduire et utiliser

Le code d'entraînement exact est archivé dans source/, avec ses empreintes dans config.json.
Le cas échéant, evaluation_source/ contient la version utilisée pour le bilan complet.
Installer PyTorch et Pillow, télécharger le dataset à la révision indiquée.
{preparation}
Puis :

```powershell
python source/train_hausa.py --version {config['version']} --dataset "CHEMIN_DATASET" --protocol "CHEMIN_PROTOCOLE" --output "CHEMIN_RUN" --test-only
```

Pour tester le package téléchargé, CHEMIN_RUN désigne son dossier racine.
Pour le bilan complet produit après une mise à jour, remplacer source/ par
evaluation_source/ dans la commande précédente.
Le modèle reste un checkpoint PyTorch personnalisé et n'est pas un modèle
Transformers chargé par AutoModel. La publication ne crée pas un serveur d'inférence.

## Limites et apport

Cette expérience documente une transcription directe Ajami–latin sous contraintes
de données et de mémoire GPU. L'association CNN–BiLSTM–CTC est établie ;
l'originalité éventuelle d'une adaptation doit être soutenue par des comparaisons
et ablations, pas par le seul nom de l'architecture ou l'utilisation d'une GPU de 6 Go.
Une seule graine ne mesure pas la variabilité entre entraînements.
{source_limit}

{measured_limit}

V13 reste une proposition : aucune amélioration ni aucun résultat V13 n'est revendiqué.
'''
    (run / "README.md").write_text(card, encoding="utf-8")


def package_run(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    config, metrics, _ = load_run(run, require_group_disjoint=True)
    write_model_card(run)
    predictions = list(csv.DictReader((run / "test_predictions.csv").open(encoding="utf-8-sig", newline="")))
    split = json.loads((run / "split.json").read_text(encoding="utf-8"))
    expected = {(row["partition"], row["file_name"]): row["transcript"] for row in split["test"]}
    if len(predictions) != len(expected) or len({(r["partition"], r["file_name"]) for r in predictions}) != len(expected):
        raise ValueError("Les predictions doivent couvrir exactement les identifiants du test fige.")
    if any(expected.get((r["partition"], r["file_name"])) != r["reference"] for r in predictions):
        raise ValueError("References du test modifiees.")
    if sum(int(r["character_errors"]) for r in predictions) != metrics["character_errors"] or sum(int(r["word_errors"]) for r in predictions) != metrics["word_errors"]:
        raise ValueError("Compteurs des predictions differents des scores publies.")
    complete = load_complete_metrics(run, config, metrics)
    if complete:
        for part, values in complete["partitions"].items():
            with (run / f"{part}_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected_rows = {(r["partition"], r["file_name"]): r["transcript"] for r in split[part]}
            if len(rows) != len(expected_rows) or {(r["partition"], r["file_name"]): r["reference"] for r in rows} != expected_rows:
                raise ValueError(f"Identifiants ou references {part} incomplets.")
            for field in ("character_errors", "word_errors", "normalized_word_errors"):
                if sum(int(r[field]) for r in rows) != values[field]:
                    raise ValueError(f"Compteur {field} incoherent pour {part}.")
    if destination.exists():
        raise FileExistsError(f"Le package existe deja : {destination}. Choisir un autre dossier.")
    destination.mkdir(parents=True)
    for name in ("best_cer.pt", "README.md", "config.json", "split.json", "vocab.json", "protocol.json", "quarantine.json", "history.csv", "test_metrics.json", "test_predictions.csv"):
        shutil.copy2(run / name, destination / name)
    shutil.copytree(run / "source", destination / "source")
    if complete:
        for name in ("metrics.json", "metrics.csv", "train_metrics.json", "validation_metrics.json", "train_predictions.csv", "validation_predictions.csv"):
            shutil.copy2(run / name, destination / name)
        shutil.copytree(run / "evaluation_source", destination / "evaluation_source")
    if load_split_audit(run, config, metrics):
        for name in ("split_audit.json", "split_audit.md", "audit_hausa_run.py"):
            shutil.copy2(run / name, destination / name)
    for name in ("source_groups.json", "excluded_public_train.json"):
        if (run/name).exists():
            shutil.copy2(run/name,destination/name)
    for name in ("EPOCHS.md", "epochs.json", "training_curves.png", "training_curves.svg"):
        if (run / name).exists():
            shutil.copy2(run / name, destination / name)
    shutil.copy2(Path(__file__).resolve().parent / "hausa_ocr" / "training_history.py", destination / "training_history.py")
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


def publish_package(folder, repo, receipt_path=None):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["author"] != AUTHOR or manifest["version"] not in ("v9", "v10", "v11", "v12"):
        raise ValueError("Attribution ou version incorrecte.")
    actual_files = {str(path.relative_to(folder)).replace("\\", "/") for path in folder.rglob("*") if path.is_file()}
    if actual_files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Le package contient des fichiers absents du manifeste ou des fichiers supplementaires.")
    for name, expected in manifest["files"].items():
        candidate = (folder / name).resolve()
        if not candidate.is_relative_to(folder.resolve()) or sha256(candidate) != expected:
            raise ValueError(f"Empreinte ou chemin incorrect : {name}")
    load_run(folder, require_group_disjoint=True)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(folder), commit_message=f"IX-World: publish verified {manifest['version']} on frozen Hausa test")
    remote = api.model_info(repo, revision=commit.oid, files_metadata=True)
    if remote.private:
        raise ValueError("Le depot Hugging Face est prive ; publication publique non confirmee.")
    siblings = {item.rfilename: item for item in remote.siblings}
    for name in actual_files:
        if name not in siblings:
            raise ValueError(f"Fichier absent du depot distant : {name}")
        data = (folder / name).read_bytes()
        item = siblings[name]
        if item.lfs is not None:
            matches = item.lfs.sha256 == hashlib.sha256(data).hexdigest()
        else:
            matches = item.blob_id == hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        if not matches:
            raise ValueError(f"Empreinte distante incorrecte : {name}")
    receipt = {"author": AUTHOR, "repository": repo, "url": f"https://huggingface.co/{repo}", "commit": commit.oid, "commit_url": commit.commit_url, "checkpoint_sha256": manifest["files"]["best_cer.pt"], "verified_files": len(actual_files)}
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    print(commit.commit_url, flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IX-World : packager puis publier uniquement un modele entraine et evalue")
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    package.add_argument("--run", type=Path, required=True)
    package.add_argument("--destination", type=Path, required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--package", type=Path, required=True)
    publish.add_argument("--repo", required=True)
    publish.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.command == "package":
        package_run(args.run, args.destination)
    else:
        publish_package(args.package, args.repo, args.receipt)
