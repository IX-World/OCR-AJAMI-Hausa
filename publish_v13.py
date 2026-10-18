"""IX-World: verify and package measured V13 results, never train the OCR."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

from hausa_ocr.metrics import UWER_DEFINITION
from hausa_ocr.publication_history import read_epoch_history, write_epoch_report
from prepare_hausa import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_v13_run(run, config, summary, audit):
    """Pure Python validation also usable by the GitHub publication workflow."""
    run = Path(run)
    if not summary or not audit or not audit["group_disjoint"]:
        raise ValueError("V13 exige un bilan complet et un audit des trois partitions.")
    split = read(run / "split.json")
    context = read(run / "context.json")
    if digest(context) != config["context_sha256"]:
        raise ValueError("Contexte V13 modifie.")
    sources = {str(row["source"]) for rows in split.values() for row in rows}
    resolved = {source: dict(context["defaults"], **context["sources"].get(source, {})) for source in sources}
    if digest(resolved) != config["resolved_context_sha256"]:
        raise ValueError("Resolution du contexte V13 incorrecte.")
    train_contexts = {str(row["source"]): resolved[str(row["source"])] for row in split["train"]}
    schema = {"schema_version": 1, "languages": [None] + sorted({r["language"] for r in train_contexts.values()} - {None}),
              "localities": [None] + sorted({r["locality"] for r in train_contexts.values()} - {None})}
    periods = [(r["year_start"], r["year_end"]) for r in train_contexts.values() if r["year_start"] is not None]
    schema.update(time_available=bool(periods), training_period_range=[min(a for a, b in periods), max(b for a, b in periods)] if periods else None)
    if schema != config["context_schema"]:
        raise ValueError("Schema contextuel non derive exclusivement du TRAIN.")
    manifest = read(run / "semantic_manifest.json")
    identity = manifest["identity"]
    annotations = read(run / "semantic_annotations.json")
    if (digest(identity) != manifest["identity_sha256"] or manifest["identity_sha256"] != config["semantic_identity_sha256"] or
            manifest["cache_sha256"] != config["semantic_cache_sha256"] or not manifest["training_only"] or not manifest["teacher_frozen"] or
            identity["train_rows_sha256"] != digest(split["train"]) or identity["train_contexts"] != train_contexts or
            identity["context_schema"] != schema or identity["annotations_sha256"] != digest(annotations) or
            identity["split_sha256"] != config["split_sha256"] or identity["vocab_sha256"] != config["vocab_sha256"] or
            identity["teacher_repo"] != config["teacher_repo"] or identity["teacher_revision"] != config["teacher_revision"]):
        raise ValueError("Provenance des cibles semantiques V13 incoherente.")
    ids = {r["partition"] + "/" + r["file_name"] for r in split["train"]}
    if set(annotations) - ids or len(annotations) != config["verified_contextual_annotations"]:
        raise ValueError("Annotations contextuelles hors TRAIN ou compteur incorrect.")
    for part, rows in split.items():
        coverage = dict(examples=len(rows), known_language=sum(resolved[str(r["source"])]["language"] is not None for r in rows),
                        known_period=sum(resolved[str(r["source"])]["year_start"] is not None for r in rows),
                        known_locality=sum(resolved[str(r["source"])]["locality"] is not None for r in rows))
        if coverage != config["context_coverage"][part]:
            raise ValueError("Couverture du contexte incorrecte.")
        values = summary["partitions"][part]
        for field in ("split_sha256", "dataset_revision", "selected_epoch", "checkpoint_sha256"):
            if values[field] != summary[field]:
                raise ValueError(f"Bilan V13 desynchronise : {part}/{field}")
        if values["evaluated_partition"] != part or values["uwer_definition"] != UWER_DEFINITION:
            raise ValueError("Role ou definition des scores incorrect.")
        with (run / f"{part}_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        expected = {r["partition"] + "/" + r["file_name"]: r for r in rows}
        if len(predictions) != len(rows) or {r["sample_id"] for r in predictions} != set(expected):
            raise ValueError("Couverture des predictions V13 incomplete.")
        for row in predictions:
            original = expected[row["sample_id"]]
            ctx = resolved[str(original["source"])]
            if (row["reference"] != original["transcript"] or str(row["source"]) != str(original["source"]) or
                    row["public_origin"] != original["partition"] or row["file_name"] != original["file_name"] or row["evaluated_partition"] != part):
                raise ValueError("Reference ou role experimental V13 modifie.")
            for field in ("language", "locality", "year_start", "year_end"):
                if row[field] != ("" if ctx[field] is None else str(ctx[field])):
                    raise ValueError("Metadonnee injectee dans une prediction differente du contexte archive.")
        for field in ("character_errors", "word_errors", "normalized_word_errors"):
            if sum(int(r[field]) for r in predictions) != values[field]:
                raise ValueError("Compteur individuel V13 incoherent.")
        for rate, numerator, denominator in (("cer", "character_errors", "reference_characters"), ("wer", "word_errors", "reference_words"), ("uwer", "normalized_word_errors", "normalized_reference_words"), ("exact_rate", "exact", "examples")):
            if not math.isclose(values[rate], values[numerator] / max(1, values[denominator]), abs_tol=1e-12):
                raise ValueError("Taux V13 incoherent avec ses compteurs.")
    with (run / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 3 or {r["partition"] for r in csv_rows} != set(split):
        raise ValueError("CSV du bilan incomplet.")
    for row in csv_rows:
        if any(float(value) != summary["partitions"][row["partition"]][field] for field, value in row.items() if field != "partition"):
            raise ValueError("CSV du bilan different des scores JSON.")
    read_epoch_history(run)
    if (run / "semantics").exists():
        if (read(run / "semantics" / "manifest.json") != manifest or read(run / "semantics" / "context.json") != context or
                read(run / "semantics" / "annotations.json") != annotations or sha256(run / "semantics" / "train_targets.pt") != config["semantic_cache_sha256"]):
            raise ValueError("Cache TRAIN du package modifie.")


def write_v13_card(run):
    from publish_hausa import load_run
    run = Path(run)
    config, test, protocol = load_run(run, require_group_disjoint=True)
    records, _ = read_epoch_history(run)
    summary = read(run / "metrics.json")
    card = """---
language:
- ha
tags:
- ocr
- ajami
- handwritten-text-recognition
- crnn
- ctc
- semantic-supervision
- IX-World
library_name: pytorch
datasets:
- IntelligenceResearchLab/Hausa
---

# Hausa Ajami OCR V13 — supervision auxiliaire AfriBERTa — IX-World

Entraînement local et publication le **18 octobre 2026**. Résultat expérimental :
**V13 ne réduit pas les erreurs de V10/V12 sur ce split. Le surapprentissage persiste.**
Les objectifs CER ≤ 20 % et WER ≤ 30 % ne sont pas atteints.

## Résultats réellement mesurés

"""
    card += "| Partition | Segments | CER | WER | UWER | Segments exacts |\n|---|---:|---:|---:|---:|---:|\n"
    for part, values in summary["partitions"].items():
        card += f"| {part} | {values['examples']} | {values['cer']:.4%} | {values['wer']:.4%} | {values['uwer']:.4%} | {values['exact']}/{values['examples']} ({values['exact_rate']:.4%}) |\n"
    card += f"""
**{len(records)} époques exécutées**, checkpoint à l'époque **{test['selected_epoch']}**, minimum
du CER de validation. Même checkpoint pour les trois ensembles, originaux sans
augmentation. Train : une seule erreur de caractère et de mot ; les erreurs restent
élevées sur les sources tenues à l'écart. [Comparaison V9–V13](COMPARISON.md).
Les résultats de test sont rapportés après entraînement ; aucun réglage n'a été
effectué pendant la publication. Leur consultation répétée dans cette campagne
devra être prise en compte pour les futures affirmations de généralisation.

## Ce que la supervision sémantique a réellement utilisé

- OCR **initialisé aléatoirement**, sans reprise des poids de V10 :
  `from_scratch=true`, `ocr_checkpoint_initialization=null`.
- Enseignant figé : [{config['teacher_repo']}](https://huggingface.co/{config['teacher_repo']}),
  révision `{config['teacher_revision']}`. Préparation sur CPU seulement ;
  aucune référence de validation ou de test n'est encodée comme cible auxiliaire.
- Moyenne des états cachés des tokens de la transcription, hors padding et tokens
  spéciaux, puis normalisation L2, dimension 768. Les offsets sont calculés avec
  un tokenizer byte-fallback vérifié contre le SentencePiece original pour chaque texte.
- Langue **haoussa renseignée pour tous les exemples**, d'après le corpus.
  Ce run n'apprend pas à détecter la langue et ne démontre pas de généralisation multilingue.
- **Aucune époque ni localité linguistique renseignée** ; entrées inconnues masquées.
  `time_available=false`, localités `[null]`. **Zéro contexte textuel complémentaire
  vérifié** : les cibles encodent les segments latins disponibles, souvent des mots isolés.
- Aucun résultat de désambiguïsation historique, régionale ou de polysémie n'est établi.
  Un encodeur contemporain ne suffit pas à démontrer cette capacité. Ni traduction,
  correction libre, enrichissement automatique continu ni auto-apprentissage en production.
- L'enseignant préentraîné apporte des connaissances externes ; seule la préparation
  de nos cibles est limitée au train. L'exposition antérieure du préentraînement aux
  textes du domaine n'a pas été auditée.

## Architecture et prétraitement exacts

**{config['parameter_count']:,} paramètres** : six convolutions gris, GroupNorm huit groupes,
ReLU, dropout spatial 0,1 après les poolings ; BiLSTM deux couches de 256 unités
par direction, dropout 0,2, séquences empaquetées selon leur longueur réelle.
Métadonnées : embeddings langue/localité de 16 dimensions chacun, branche temporelle
de 16 dimensions, modulation FiLM bornée par 0,1. Dropout séquentiel 0,2 puis
classification CTC à {config['classes']} classes (BLANK=0, UNK=1).
Branche auxiliaire : attention sur les pas valides, projection 512→768 et LayerNorm.
Elle supervise l'encodeur pendant l'entraînement ; elle ne réordonne pas les sorties.

Hauteur 96, largeur proportionnelle, Lanczos, miroir horizontal RTL, un canal
gris normalisé dans [-1,1], padding blanc +1. **Pas de canal d'encre locale**.
Décodage CTC glouton, répétitions fusionnées et BLANK supprimé, sans beam search,
lexique ni modèle de langue au décodage. Aucune détection des lignes sur page complète.
La tâche est image Ajami → texte latin, sans annotation Ajami intermédiaire.

## Données, split et normalisation

Dataset `IntelligenceResearchLab/Hausa`, révision figée `{config['dataset_revision']}`.
**4 972 train / 200 validation / 401 test**, 5 573 exemples valides. Split par manuscrit
et groupe de copiste connu, sources 1 et 2 regroupées, mêmes identifiants et vocabulaire
que V9–V12. Toutes les sources de validation/test sont absentes du train.
Le test a été reconstruit sur le corpus complet ; il n'est pas le test public initial de 400 images.
Trois chemins corrigés, annotation répétée identique retirée ; les deux images identiques
`a_ajami_03400.png` et `a_ajami_03401.png` aux références contradictoires restent exclues.
Les quasi-doublons et les copistes non identifiés restent à contrôler.

Split SHA-256 : `{config['split_sha256']}`.
Vocabulaire construit sur le train seul : `{config['vocab_sha256']}`.
CER/WER stricts conservent casse, espaces et diacritiques NFC. Calcul global :
distances d'édition cumulées divisées par les longueurs de référence cumulées.
UWER : {UWER_DEFINITION}
Les compteurs bruts, dénominateurs et prédictions sont tous publiés.

## Entraînement sur GTX 1060 6 Go

- Huit vues fixes par image train : original, luminosité/contraste, épaisseur d'encre,
  résolution, flou, rotation, géométrie et combinaison modérée ; paramètres exacts
  et graines dans `source/hausa_ocr/augmentations.py`. **39 776 vues par époque**.
  Validation et test sans augmentation. Les vues ne sont pas des images indépendantes.
- Float32 ; microbatch jusqu'à 6 exemples et 600 000 pixels après padding ; accumulation
  de 4 microbatches (jusqu'à 24 exemples par mise à jour), workers 0, groupes de largeur.
- AdamW LR 0,0005, weight decay 0,001, clipping de norme 5 ; ReduceLROnPlateau
  sur CER validation, facteur 0,5, patience 3, minimum 0,000001.
- Perte = CTC + λ × (0,5 × cosinus + 0,5 × contrastive). Température contrastive 0,1,
  positifs multiples pour un même texte/contexte, négatifs de même langue dans le
  microbatch réel seulement. L'accumulation n'agrandit pas le groupe contrastif.
- λ=0 aux époques 1–3, puis 0,02 à 0,10 des époques 4 à 8 ; ensuite λ=0,10.
- Maximum 100, minimum 30 époques, patience 15 avec seuil 0,0005 pour l'arrêt.
  Sélection des poids au minimum absolu du CER validation. Graine 42 ; cuDNN benchmark,
  pas de garantie de déterminisme bit à bit. Une seule graine mesurée.
- PyTorch {config['torch']}, CUDA {config['cuda']}, Python {config['python']}.
- Durée enregistrée train + validation : **{test['elapsed_training_seconds']/3600:.2f} h**,
  hors préparation de l'enseignant, sauvegardes et bilan final ; pic **{test['peak_vram_mib']:.1f} MiB
  alloués par PyTorch**, pas la mémoire totale occupée/réservée sur le GPU.

V13 change aussi normalisation, dropout, empaquetage, LR, régularisation et batching.
Cette comparaison n'isole donc pas l'effet de la seule perte sémantique. Une ablation
λ=0 avec la même architecture et plusieurs graines reste à réaliser avant toute
conclusion causale. Aucune nouvelle expérience n'a été lancée pour cette publication.

## Historique complet et intégrité

[Toutes les époques et unités](EPOCHS.md) · [CSV brut](history.csv) · [JSON numérique](epochs.json)
· [Bilan train/val/test](metrics.csv) · [Vérifications](verification.json).
La loss combinée, CTC, cosinus, contrastive, λ et le nombre de mises à jour sont
publiés séparément. Les CER/WER/UWER train et test ne sont pas mesurés à chaque époque.

![Courbes enregistrées de V13](training_curves.png)

[Courbes vectorielles](training_curves.svg). Aucun lissage ni score reconstitué.
Code d'entraînement **et d'évaluation** dans `source/` (même version), empreintes
dans `config.json`, commit `{config['source_commit']}`.
Checkpoint SHA-256 : `{test['checkpoint_sha256']}`.
Les manifestes et cibles auxiliaires du TRAIN sont dans `semantics/` ; SHA-256 du cache
`{config['semantic_cache_sha256']}`. Les poids AfriBERTa ne sont pas redistribués.
Les images du corpus ne sont pas incluses dans le package.

## Utilisation locale sans réentraîner

Télécharger le package complet HF ou le ZIP de la release. Installer PyTorch et Pillow
compatibles avec votre machine ; les versions d'entraînement sont indiquées ci-dessus.
L'enseignant AfriBERTa et Transformers ne sont **pas nécessaires** à l'inférence.

```cmd
python infer_v13.py --model-dir "CHEMIN_PACKAGE" --image "CHEMIN_IMAGE.png" --source 15 --device cpu
```

`--source` doit désigner la provenance documentaire connue dans le contexte archivé ;
il ne représente pas un label prédit. Ici tous les contextes sont identiques (haoussa,
époque/localité inconnues). La prédiction peut varier légèrement avec le padding d'un
lot, car GroupNorm agrège aussi les positions spatiales : les résultats publiés utilisent
les microbatches de validation/test archivés et les dimensions originales du split.

Pour recalculer le bilan des trois partitions, sans entraînement (cela utilise le test) :

```cmd
python source\\train_v13.py --dataset "D:\\ai_test\\Hausa_repo_2026-10-15" --protocol "CHEMIN_PACKAGE" --context "CHEMIN_PACKAGE\\context.json" --semantics "CHEMIN_PACKAGE\\semantics" --output "CHEMIN_PACKAGE" --evaluate-only --device cpu
```

Pour un nouvel entraînement indépendant à partir de zéro, remplacer `--evaluate-only`
par `--fresh-start` et utiliser **un nouveau dossier --output**, les mêmes options et
la même révision du dataset. Sous Windows, les raccourcis du dépôt sont décrits dans
`source/README_V13.md`. Les poids ne sont pas un AutoModel Transformers.
La publication est un téléchargement de modèle ; elle ne crée pas de serveur d'inférence.

## Attribution et limites

Code, adaptation et entraînement : **IX-World**.
Projet : https://github.com/IX-World/OCR-AJAMI-Hausa
Les manuscrits, annotations, crédits et droits du corpus restent ceux de leurs auteurs
et du dataset d'origine. L'enseignant garde ses crédits et conditions propres.
Cette fiche ne crée pas de nouvelle licence pour ces ressources.
L'architecture CNN–BiLSTM–CTC est établie ; ni une GPU de 6 Go ni l'ajout d'une
perte sémantique ne suffisent à établir une nouveauté scientifique ou une amélioration.
Le modèle nécessite une relecture humaine ; ces résultats ne montrent pas une
transcription autonome fiable.
"""
    card += "\n" + (run / "split_audit.md").read_text(encoding="utf-8")
    (run / "README.md").write_text(card, encoding="utf-8", newline="\n")


def package_v13(run, semantics, destination):
    from publish_hausa import load_run
    run, semantics, destination = map(lambda p: Path(p).resolve(), (run, semantics, destination))
    config, _, _ = load_run(run, require_group_disjoint=True)
    if config["version"] != "v13" or sha256(semantics / "train_targets.pt") != config["semantic_cache_sha256"]:
        raise ValueError("Cache TRAIN different de celui utilise par V13.")
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise FileExistsError("Le package existe deja ; ne pas ecraser un artefact publie.")
    write_epoch_report(run)
    write_v13_card(run)
    destination.mkdir(parents=True)
    names = ("best_cer.pt", "config.json", "context.json", "semantic_manifest.json", "semantic_annotations.json", "split.json", "vocab.json", "protocol.json", "quarantine.json", "source_groups.json", "excluded_public_train.json", "history.csv", "metrics.json", "metrics.csv", "README.md", "split_audit.json", "split_audit.md", "audit_hausa_run.py", "EPOCHS.md", "epochs.json", "training_curves.png", "training_curves.svg")
    for name in names:
        shutil.copy2(run / name, destination / name)
    for part in ("train", "validation", "test"):
        for suffix in ("metrics.json", "predictions.csv"):
            shutil.copy2(run / f"{part}_{suffix}", destination / f"{part}_{suffix}")
    shutil.copytree(run / "source", destination / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "semantics").mkdir()
    for name in ("train_targets.pt", "manifest.json", "context.json", "annotations.json"):
        shutil.copy2(semantics / name, destination / "semantics" / name)
    for source, name in ((Path(__file__), "publish_v13.py"), (Path(__file__).parent / "infer_v13.py", "infer_v13.py"), (Path(__file__).parent / "hausa_ocr" / "publication_history.py", "training_history.py")):
        shutil.copy2(source, destination / name)
    load_run(destination, require_group_disjoint=True)
    return destination


def finalize_package(destination):
    destination = Path(destination)
    archive = destination.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    manifest = {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file() and p != destination / "manifest.json"}
    write_json(destination / "manifest.json", {"author": "IX-World", "version": "v13", "files": dict(sorted(manifest.items()))})
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(destination).as_posix())
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IX-World : packager V13 termine avec son historique et ses cibles TRAIN.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    folder = package_v13(args.run, args.semantics, args.destination)
    print(finalize_package(folder))
