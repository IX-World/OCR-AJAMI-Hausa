# V13 — supervision sémantique contextuelle, IX-World

V13 apprend directement à transcrire une image Ajami vers le haoussa latin.
Il ajoute une supervision auxiliaire issue d'un encodeur textuel préentraîné
figé. La langue, l'époque et la localité documentées conditionnent le réseau.
Le code conserve exactement le protocole commun **4 972 train / 200 validation /
401 test**, sans reprendre de checkpoint OCR V9–V12.

## Résultat du 18 octobre 2026

V13 a terminé **49 époques**, checkpoint retenu à l'époque **34**.
Le test donne **CER 39,0987 % / WER 86,3060 % / UWER 86,0136 %**.
Le surapprentissage persiste ; ces résultats sont moins bons que V10 et V12.
Ce run est parti de poids OCR aléatoires. La langue est renseignée comme haoussa,
mais toutes les époques/localités restent inconnues et les branches correspondantes
sont masquées ; aucun contexte textuel complémentaire n'a été annoté.

[Modèle HF et fiche complète](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v13-20261015)
· [Release et poids](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v13-hausa-all-grouped-20261018)
· [Comparaison V9–V13](models/v13_all_grouped_20261018/COMPARISON.md)
· [Toutes les époques](models/v13_all_grouped_20261018/EPOCHS.md).

Le pic enregistré est de **892,3 MiB alloués par PyTorch**, hors mémoire totale
réservée/occupée sur la carte ; durée train + validation enregistrée : **7,86 h**.
Le package inclut un script `infer_v13.py` qui utilise les sources archivées,
sans enseignant. La préparation et les commandes ci-dessous documentent le run
initial : conserver le dossier existant et choisir un nouvel `--output` pour
une nouvelle expérience.

## Commande CMD depuis D:\ai_test

Les raccourcis sont installés à la racine de `D:\ai_test`.
La préparation télécharge environ 446 Mo de poids AfriBERTa, utilise le CPU et
ne change pas l'environnement Python d'entraînement existant. Les bibliothèques
de l'encodeur sont installées dans le dossier isolé `.v13-semantic-deps` du dépôt.

```cmd
cd /d "D:\ai_test"
call preparer_v13.cmd && call lancer_v13.cmd --epochs 100 --min-epochs 30 --patience 15 --batch-size 6 --accumulate 4 --max-batch-pixels 600000
```

La première commande prépare les cibles du train. L'entraînement ne démarre
que si cette préparation réussit. À la fin, le script recharge le checkpoint
au **minimum du CER de validation**, puis mesure CER, WER, UWER et taux exact
sur train, validation et test, avec une vue originale par image.

Un run déjà présent n'est jamais écrasé. Sortie par défaut :
`D:\ai_test\runs_hausa_all_grouped_20261016\v13_seed42`.
Une interruption conserve le dernier checkpoint d'époque complète :

```cmd
call reprendre_v13.cmd
```

Pour réévaluer un V13 terminé, sans réentraîner :

```cmd
call tester_v13.cmd
```

Conserver les arguments du lancement initial pour une reprise ou une
réévaluation si des réglages ont été modifiés. Aucun ancien checkpoint OCR
ne sert d'initialisation ; la reprise accepte seulement le run V13 concerné.

## Réseau et perte

- Entrée en niveaux de gris, hauteur 96, proportions préservées, miroir RTL,
  valeurs `2I-1` et remplissage blanc, comme la branche visuelle V10.
- CNN à six convolutions (64, 128, 256, 256, 384, 384), GroupNorm à huit groupes
  pour les petits microbatches, dropout spatial 0,1 après les max-poolings.
- BiLSTM de deux couches, 256 unités par direction, dropout 0,2. Les séquences
  sont empaquetées selon leur longueur réelle ; les tokens de remplissage ne
  participent ni au parcours récurrent ni au pooling sémantique.
- Un conditionnement résiduel FiLM, borné à 0,1, utilise la langue et la localité
  (embeddings de 16 dimensions chacun), ainsi qu'un intervalle temporel encodé
  par son milieu et sa durée. La source du manuscrit n'est pas une entrée du réseau.
- Branche CTC : projection 512 vers les 80 classes du vocabulaire train archivé.
- Branche auxiliaire : attention sur les positions valides puis projection
  512 vers 768, pour aligner l'image sur les représentations textuelles.
- Perte : `CTC + lambda * (0.5 * cosine + 0.5 * contraste)`.
  Le contraste est calculé entre textes de même langue dans le microbatch,
  température 0,1. Des entrées textuelles identiques, y compris plusieurs vues
  d'une même image, sont des positifs. Une même graphie dans deux contextes
  attestés différents est exclue des négatifs contrastifs ; chaque exemple
  conserve sa cible contextualisée pour l'alignement cosinus. Cela évite
  d'imposer arbitrairement un sens unique aux mots polysémiques.
  C'est un alignement de représentations contextuelles ; il ne prouve pas
  l'acquisition du sens linguistique.
- `lambda` vaut 0 aux trois premières époques, puis monte linéairement de 0,02
  à 0,10 entre les époques 4 et 8. Réglages modifiables et archivés.

V13 change aussi la normalisation, le dropout et la gestion des longueurs par
rapport à V10. Pour isoler l'apport sémantique, comparer à **V13 avec lambda=0**
dans un autre dossier, sur le même split et avec les mêmes réglages :

```cmd
call lancer_v13.cmd --semantic-weight 0 --output "D:\ai_test\runs_hausa_all_grouped_20261016\v13_sans_semantique_seed42"
```

Ce comparatif reste une expérience à réaliser ; le test ne doit pas servir
à choisir lambda, la patience ou les autres hyperparamètres.

## Encodeur textuel et séparation des données

Encodeur : [`castorini/afriberta_base`](https://huggingface.co/castorini/afriberta_base),
révision figée `95b703f498c9cee56be5f5bbc5140022dc86099e`.
Son préentraînement inclut notamment le haoussa et le swahili. Le présent
dataset fournit une supervision haoussa ; le code ne prétend pas évaluer
d'autres langues ni détecter automatiquement la langue d'un mot isolé.
Une langue non prise en charge par cet encodeur est refusée ; ajouter un nom
de langue au JSON ne crée pas un nouveau modèle linguistique.

Les poids de l'encodeur sont conservés, figés et utilisés seulement sur CPU
pour préparer les références **du train**. Ils ne sont pas chargés sur le GPU
pendant l'OCR. L'encodeur fonctionne en mode évaluation, sans dropout ni gradient.
Le checkpoint public `.bin` est lu avec `torch.load(weights_only=True)`, son
SHA-256 est vérifié et aucune exécution de code distant n'est autorisée.

Les cibles sont les moyennes normalisées des tokens de la référence, hors
tokens spéciaux et padding. Si un contexte textuel vérifié est disponible,
seuls les tokens correspondant à la référence sont agrégés, après encodage
dans ce contexte. Un contexte trop long déclenche une erreur plutôt qu'une
troncature silencieuse. Aucun texte de validation/test ne prépare ces cibles.
Le tokenizer rapide active explicitement le repli sur octets de SentencePiece,
nécessaire pour certains diacritiques et lettres du corpus. Chaque séquence
est comparée à la tokenisation originale avant encodage ; aucune simplification
des transcriptions n'est utilisée pour contourner un écart.

Le cache `D:\ai_test\v13_semantics\train_targets.pt` et son manifeste fixent
les identifiants du train, les annotations, le code de préparation et le modèle.
Les cibles déjà préparées sont réutilisées si leur identité correspond.
Une modification nécessite un **nouveau dossier** de préparation et de run :

```cmd
call preparer_v13.cmd --output "D:\ai_test\v13_semantics_contexte_v2"
call lancer_v13.cmd --semantics "D:\ai_test\v13_semantics_contexte_v2" --output "D:\ai_test\runs_hausa_all_grouped_20261016\v13_contexte_v2_seed42"
```

## Langue, époque et localité : ne pas inventer les annotations

`D:\ai_test\v13_context.json` est installé depuis `configs/v13_context.json`
sans écraser un fichier personnel existant. Par défaut : langue `ha`, époque
et localité `null`. Le corpus est documenté comme haoussa ; les origines des
annotateurs ou le lieu de conservation ne prouvent pas une localité linguistique.
**Avec ces valeurs par défaut, V13 bénéficie d'une supervision textuelle
haoussa contemporaine, mais n'apprend pas de distinctions historiques ou locales.**

Pour une source dont la provenance est réellement établie, ajouter une entrée
à `sources`. Le schéma ci-dessous est un exemple de structure, pas une annotation
historique à recopier comme vérité :

```json
"sources": {
  "ID_SOURCE_A_VERIFIER": {
    "language": "ha",
    "year_start": 1800,
    "year_end": 1899,
    "locality": "REGION_LINGUISTIQUE_DOCUMENTEE",
    "evidence": "Notice d'archive, page, identifiant et justification"
  }
}
```

Les années représentent une période de composition en ère commune (`CE`),
éventuellement incertaine ; distinguer date de copie et date de composition.
Les valeurs inconnues restent `null`. La localité est une région linguistique
documentée, sans affectation automatique du sens selon un pays actuel.
Les vocabulaires de langue/localité sont ajustés **sur le train seulement**.
Une catégorie jamais vue en train est traitée comme inconnue à l'évaluation.
La période est continue ; une date fournie ne garantit pas la généralisation
à une époque absente du train. Si aucune date n'est connue dans le train,
la branche temporelle est aussi neutralisée à l'évaluation, même si une date
de validation/test a été renseignée : elle ne doit pas exploiter des poids
temporels restés sans supervision.

Pour enrichir les représentations contextuelles avec des phrases attestées,
fournir un fichier JSONL réservé aux exemples du train :

```json
{"sample_id":"train/NOM_IMAGE_DU_TRAIN.png","verified":true,"evidence":"Source de la phrase, page et relecteur","context_before":"passage precedent","context_after":"passage suivant"}
```

La référence du split est insérée entre ces deux passages, sans être modifiée.
Les identifiants de validation/test, doublons et contextes non vérifiés sont
refusés. `train/` ou `test/` dans l'identifiant désigne le dossier public
d'origine : c'est le rôle expérimental archivé qui décide de l'admissibilité.

```cmd
call preparer_v13.cmd --annotations "D:\ai_test\v13_annotations_train.jsonl" --output "D:\ai_test\v13_semantics_annote"
call lancer_v13.cmd --semantics "D:\ai_test\v13_semantics_annote" --output "D:\ai_test\runs_hausa_all_grouped_20261016\v13_annote_seed42"
```

Les contextes attestés doivent rester dans des sources d'apprentissage autorisées,
sans importer les passages des manuscrits exclus du train. Les exemples isolés
ne permettent pas d'apprendre seuls la polysémie d'une phrase. La compréhension
historique/localisée demanderait des attestations et une évaluation linguistique
distincte, au-delà des CER/WER/UWER.

## Recette GTX 1060 6 Go

- Float32 ; pas de flash attention, BF16 ou AMP FP16 imposé à la carte Pascal.
- Microbatch maximal 6, accumulation de quatre microbatches, soit au maximum
  24 images par mise à jour. Le dernier groupe est pondéré par son effectif réel.
- Budget de 600 000 pixels remplis par microbatch, regroupement par largeur.
  Aucune fin de transcription n'est coupée pour respecter ce budget.
- Huit augmentations fixes du train héritées de V10 : original, luminosité/
  contraste, épaisseur d'encre, résolution, flou, rotation, échelle et combinaison.
  **39 776 vues par époque** ; les partitions et leurs images réelles ne changent pas.
- AdamW, LR `0.0005`, weight decay `0.001`, gradient clip 5.
  Réduction du LR sur le CER de validation, facteur 0,5, patience 3, minimum `1e-6`.
- Maximum 100 époques ; arrêt précoce après au moins 30 époques, patience 15,
  seuil de progression significative `0.0005` en CER. Le meilleur checkpoint est
  néanmoins le minimum strict du CER observé.
- `num_workers=0` pour éviter les problèmes de multiprocessing sous Windows.
  Les microbatches augmentent le nombre de lignes de progression ; le nombre
  d'images et de vues reste explicite. La durée doit être mesurée localement.

Si la carte est libre mais que la mémoire manque, nouveau lancement :

```cmd
call lancer_v13.cmd --batch-size 4 --accumulate 6 --max-batch-pixels 400000 --output "D:\ai_test\runs_hausa_all_grouped_20261016\v13_batch4_seed42"
```

Le budget est une limite de pixels, pas une garantie sur toute la mémoire CUDA.
Ne pas lancer V13 simultanément avec un autre entraînement sur les 6 Go.

## Résultats et traçabilité

Chaque époque affiche les pertes CTC/sémantique, lambda, CER/WER/UWER de
validation, taux exact, pente CER, gain, LR, temps, pic VRAM et cinq exemples.
Les fichiers `history.csv`, `best_cer.pt`, `last.pt`, `config.json` et `source/`
permettent de retrouver les réglages et le code exécuté.
Le bilan final produit `metrics.csv`, `metrics.json`, les trois fichiers
`*_metrics.json`, les trois fichiers `*_predictions.csv` et un `README.md` du run.
Le test n'est évalué qu'après la sélection par validation.

UWER conserve sa définition commune : casefold Unicode, suppression des marques
diacritiques `Mn` après NFD, retour NFC, séparation par espaces ; ponctuation et
lettres haoussa non décomposées conservées. Les CER/WER restent stricts.

Les poids de l'OCR sont initialisés aléatoirement, pas tous à zéro. Seules les
entrées réservées aux catégories inconnues et les masques suivent une convention
neutre. Le modèle de langue apporte son préentraînement externe documenté.
L'encodeur est auxiliaire à l'entraînement ; l'inférence CTC de V13 n'a besoin
ni de références ni d'AfriBERTa. Il n'y a pas d'auto-apprentissage à partir de
prédictions non relues, de normalisation moderne imposée, ni de publication
automatique sur HF/GitHub. Une nouvelle version linguistique nécessite une
préparation et une expérience distinctes.

Références : [AfriBERTa](https://huggingface.co/castorini/afriberta_base),
[représentations temporelles et géographiques](https://aclanthology.org/2020.conll-1.1/).

## Contrôles du 17 octobre 2026

35 tests CPU passent, dont le passage des gradients sémantiques au CNN, les
masques de contexte inconnu, la gestion des graphies polysémiques, le budget
de pixels et un cycle synthétique entraînement/sauvegarde/évaluation/rechargement.
Les 2 629 références distinctes du train reproduisent la tokenisation originale.
Les 4 972 cibles réelles ont été préparées et vérifiées (768 dimensions chacune),
sans références de validation/test dans l'encodeur. Le contrôle sur les chemins
réels et une inférence de validation CPU passent. V13 compte 6 553 393 paramètres
avec le contexte par défaut. Aucun entraînement OCR du corpus n'a été lancé.
La GTX 1060 étant occupée par un autre entraînement lors de ces contrôles,
le pic mémoire et les performances V13 sur GPU restent à mesurer localement.
