---
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

# Hausa Ajami OCR — V12 — contribution IX-World

## Attribution et portée

Code, adaptation expérimentale et entraînement : **IX-World**.
Phase : `from_scratch`.
Dépôt du projet : https://github.com/IX-World/OCR-AJAMI-Hausa
Commit de code enregistré : `c4f91a496c11321d5b24f32d3d22021a5243f1bf`.
Le corpus provient d'IntelligenceResearchLab/Hausa ; sa provenance, ses crédits
et ses conditions de réutilisation restent ceux du dépôt d'origine.
Cette fiche ne réattribue pas les manuscrits ni les annotations à l'auteur du code.

Le modèle transforme une image de segment manuscrit haoussa Ajami en texte latin.
Il n'effectue pas de détection sur une page complète et n'exige aucune
transcription intermédiaire en caractères Ajami.

## Version des données et séparation des ensembles

- Révision du dataset : `1fb85fa070b6cf6432f1d45d21f78851d57f8944`.
- Entraînement : **4972 segments réels**.
- Validation : **200 segments**, sélectionnés selon la règle ci-dessous.
- Test : **401 exemples**, sélectionnés par groupes sur le corpus complet ; le test public initial n'est pas conservé comme partition.
- SHA-256 du split : `fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5`.
- Construction de la validation : groupes entiers au plus près de 200 exemples de validation, puis de 400 exemples de test ; toutes les autres images sont entraînées ; graine de référence 42.
- Les trois chemins invalides ont été corrigés ; la référence de « ranar »
  a été rétablie vers ajami_01289.png ; le doublon d'annotation identique a été supprimé.
- Les deux images identiques a_ajami_03400/03401, annotées « da »/« mutane »,
  sont exclues sur décision de l'utilisateur et conservées dans quarantine.json.
- Aucun doublon de pixels exact entre train, validation et test n'est présent.
  Les quasi-doublons et l'indépendance des copistes ne sont pas garantis.

Ce protocole diffère du split historique 2 942/96/160 : les scores de cette
campagne ne constituent pas une comparaison directe avec les anciens scores.
Tous les nouveaux V9 à V12 utilisent exactement le même split et vocabulaire.

Tous les exemples valides du corpus HF sont répartis entre les trois ensembles, sans manuscrit partagé. Les sources 1 et 2 sont regroupées par copiste connu. Seules les deux annotations contradictoires restent en quarantaine. L'identité des autres copistes n'est pas documentée.

## Architecture et prétraitement

9 convolutions 3×3, BatchNorm et ReLU ;
deux premiers poolings 2×2, puis réduction verticale ; BiLSTM de deux couches,
256 unités par direction, dropout 0,2 ; projection linéaire et CTC.
Nombre de paramètres : **7,621,840**.
Nombre de classes : **80**, BLANK et UNK compris.
Dropout convolutionnel : 0.05.

Hauteur 96 pixels, largeur proportionnelle, interpolation Lanczos,
miroir horizontal RTL ; niveaux de gris ramenés à [-1,1].
Nombre de canaux : 2.
Pour V11/V12, second canal d'encre locale : fond flouté avec rayon 5 pixels,
I = clip(1,6 × max(fond − gris, 0), 0, 1), puis 2I−1.
Padding gris blanc (+1), et absence d'encre (−1) pour le second canal.
Le BiLSTM reçoit des séquences complétées, sans empaquetage.

Le vocabulaire est construit exclusivement sur le train ; les références
d'évaluation sont conservées en Unicode NFC, avec casse, espaces et diacritiques.
Caractères absents du train : `{"validation": [], "test": ["'", "1", "[", "]"]}`.
Une classe UNK représente les caractères inconnus ; aucune référence de test
n'est réécrite pour faciliter le calcul du CER/WER.

## Entraînement réellement configuré

- Initialisation aléatoire sans préentraînement ; graine 42.
- Augmentation : `fixed_8_views` ; 39776 vues par époque.
- Les vues V10/V11/V12 sont fixes ; V9 applique l'augmentation dynamique à 30 % des segments.
- Batch 20, groupes de largeur 160, workers 0.
- AdamW, LR initial 0.001, weight decay 0,0001, clipping de norme 5.
- ReduceLROnPlateau : CER validation, facteur 0,5, patience 3, minimum 0,000001.
- Maximum 100 époques ; minimum 30 avant arrêt ;
  patience 15, amélioration minimale 0,0005 pour la patience.
- Sélection du checkpoint par minimum absolu du CER de validation.
- Époques effectivement exécutées : 65 ; checkpoint publié : époque 50.
- Float32 ; cuDNN benchmark ; déterminisme strict non imposé.
- GPU : NVIDIA GeForce GTX 1060 6GB ; PyTorch 2.5.1+cu121 ; CUDA 12.1 ;
  Python 3.12.14 ; Pillow 12.3.0.

Les adaptations communes de cette campagne sont décrites dans config.json,
notamment le vocabulaire train seul, le padding, le regroupement en lots
et l'arrêt anticipé harmonisé. Il ne s'agit pas d'une reproduction bit à bit
des anciens scripts à chemins C:\ai-test.

## Résultats mesurés, sans résultat anticipé

| Partition | Segments | CER | WER | UWER | Segments exacts |
|---|---:|---:|---:|---:|---:|
| train | 4972 | 0.0000% | 0.0000% | 0.0000% | 100.0000% |
| validation | 200 | 41.9691% | 84.1163% | 81.4318% | 21.5000% |
| test | 401 | 33.5425% | 79.6784% | 78.8499% | 1.2469% |

Les trois partitions sont évaluées avec **le même checkpoint**, sélectionné
sur le CER de validation. Le train est réévalué sur chaque image originale
une seule fois, sans augmentation ; son score mesure l'ajustement aux données vues.
Les scores de validation interviennent dans la sélection ; seul le test reste tenu à l'écart.

Les compteurs bruts et normalisés, les dénominateurs et les empreintes sont
dans metrics.json et dans les trois fichiers *_metrics.json. Le tableau est
également exporté dans metrics.csv ; les prédictions sont dans train_predictions.csv,
validation_predictions.csv et test_predictions.csv. Le code d'évaluation est
archivé séparément dans evaluation_source/, avec ses empreintes dans metrics.json.

Définition de l'UWER dans ce projet : Unicode casefold, NFD, suppression des marques Mn, puis NFC et separation des mots par les espaces ; ponctuation et lettres haoussa non decomposees conservees.
L'UWER ne remplace pas le WER strict ; la ponctuation reste évaluée.


## Historique de toutes les époques

Les **65 époques enregistrées** sont disponibles dans
[le tableau complet avec les unités](EPOCHS.md), le [CSV brut](history.csv)
et le [JSON numérique](epochs.json). Le tableau indique les meilleurs scores
de validation, le checkpoint retenu, la loss train, les CER/WER/UWER et le taux
exact de validation, le LR utilisé et suivant, le temps, la VRAM, la pente
et le gain du CER. Les courbes sont incluses lorsqu'elles ont été générées.
Les CER/WER/UWER train et test ne sont pas mesurés à chaque époque : leur bilan
final utilise le checkpoint retenu sur la validation.

![Courbes des mesures enregistrées pendant les époques](training_curves.png)

[Version vectorielle SVG](training_curves.svg).

| Provenance et ressources | Valeur |
|---|---:|
| Erreurs de caractères / caractères de référence | 3260 / 9719 |
| Erreurs de mots / mots de référence | 1635 / 2052 |
| Checkpoint sélectionné à l'époque | 50 |
| Durée enregistrée d'entraînement | 1058.40 min |
| Pic VRAM allouée | 5108.3 MiB |

Le CER et le WER sont des rapports globaux des distances d'édition cumulées.
Les mots sont séparés par les espaces. Décodage glouton CTC, fusion des
répétitions et suppression de BLANK ; aucun lexique ni correcteur.
Les prédictions individuelles et les compteurs sont dans test_predictions.csv.
SHA-256 du checkpoint : `99a0d0f2c824a242b1b5e4864a5af2f4df0c291064ad0f1eba2ddeb6d5c3f6c1`.

## Audit du split et interpretation de l'ecart validation/test

Audit du checkpoint `99a0d0f2c824a242b1b5e4864a5af2f4df0c291064ad0f1eba2ddeb6d5c3f6c1` et du split `fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5`.

Les exemples et leurs references couvrent exactement le corpus corrige, en tenant compte des exclusions documentees. Aucun identifiant ni hash de pixels exact n'est partage entre partitions. La recherche de quasi-doublons n'a pas ete realisee. Les CER, WER et UWER ont ete recalcules depuis les predictions et leurs references : ils correspondent aux fichiers de scores.

| Composition | Train | Validation | Test |
|---|---:|---:|---:|
| Segments | 4972 | 200 | 401 |
| Mots | 10657 | 447 | 2052 |
| Mots moyens par segment | 2.14 | 2.23 | 5.12 |
| Segments d'un seul mot | 3777 | 156 | 126 |
| Mots stricts absents du train (occurrences) | 0 | 144 | 505 |
| References completes deja presentes dans le train | 4972 | 89 | 65 |
| Segments de sources absentes du train | 0 | 200 | 401 |

La validation contient 78.00% de segments d'un seul mot, contre 31.42% au test. 401 exemples de test proviennent de sources absentes du train ; la repartition complete par source est dans split_audit.json. Ces differences de composition doivent etre prises en compte pour interpreter les scores de validation et du test du protocole.

| Groupe | Validation N / mots | Validation WER | Test N / mots | Test WER |
|---|---:|---:|---:|---:|
| Source presente au train | 0 / 0 | non mesure | 0 / 0 | non mesure |
| Source absente du train | 200 / 447 | 84.12% | 401 / 2052 | 79.68% |
| Segments d'un mot | 156 / 156 | 73.72% | 126 / 126 | 100.79% |
| Segments de 2 a 4 mots | 2 / 5 | 100.00% | 12 / 43 | 109.30% |
| Segments de 5 mots et plus | 42 / 286 | 89.51% | 263 / 1883 | 77.59% |
| Reference complete vue au train | 89 / 89 | 56.18% | 65 / 65 | 92.31% |
| Reference complete non vue au train | 111 / 358 | 91.06% | 336 / 1987 | 79.27% |

Ces groupes decrivent des associations, sans isoler une cause unique. La longueur ne majore pas mecaniquement le WER, qui est normalise par le nombre de mots ; elle change notamment les conditions visuelles de reconnaissance. La validation intervient aussi dans la selection du checkpoint et ne constitue pas un second test independant. Le WER des sources connues et inconnues du test doit etre examine avant d'attribuer l'ecart aux seules sources. Les conventions d'annotation et la qualite visuelle n'ont pas ete relues integralement dans cet audit.

Le split et le checkpoint documentes restent figes pour comparer V9 a V12. Les decisions de repartition reposent sur les groupes de provenance et leurs effectifs. Le fichier de protocole indique si le test public initial est conserve ou reconstruit.


## Reproduire et utiliser

Le code d'entraînement exact est archivé dans source/, avec ses empreintes dans config.json.
Le cas échéant, evaluation_source/ contient la version utilisée pour le bilan complet.
Installer PyTorch et Pillow, télécharger le dataset à la révision indiquée.
Appliquer prepare_hausa.py pour les corrections initiales, puis prepare_hausa_grouped.py pour la séparation par provenance. Les fichiers source_groups.json et excluded_public_train.json décrivent le nouveau protocole. Utiliser impérativement --all-images pour retrouver ce split.
Puis :

```powershell
python source/train_hausa.py --version v12 --dataset "CHEMIN_DATASET" --protocol "CHEMIN_PROTOCOLE" --output "CHEMIN_RUN" --test-only
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
Aucune source n'est partagée entre les partitions ; les références lexicales peuvent se répéter. L'indépendance des copistes non identifiés et l'absence de quasi-doublons restent à vérifier.

Le WER du train (0.00%) et celui du test (79.68%) montrent un écart de généralisation. Le WER de test est un taux d'erreur, pas un taux de réussite. Ce checkpoint est publié comme résultat expérimental reproductible ; ces scores ne démontrent pas une transcription autonome fiable.

V13 dispose d'un code expérimental distinct ; aucun résultat ni aucune amélioration V13 n'est revendiqué avant son évaluation.

## Publication du 18 octobre 2026 et comparaison

Entraînement terminé le 17 octobre ; 65 époques, checkpoint publié à l'époque 50. V12 n'utilise pas l'encodeur sémantique V13. Le chargement strict et 20 prédictions de validation sur CPU ont été vérifiés.

### Comparaison V9–V12 sur le même split — IX-World

Protocole commun : 4 972 train / 200 validation / 401 test, mêmes identifiants et vocabulaire, graine 42. Chaque checkpoint est sélectionné sur son CER de validation.

| Modèle | Époque retenue | Époques exécutées | Test CER | Test WER | Test UWER | Segments exacts |
|---|---:|---:|---:|---:|---:|---:|
| V9 | 33 | 48 | 40.6935% | 86.9396% | 86.1598% | 1.2469% |
| V10 | 39 | 54 | 34.8801% | 79.7758% | 79.2398% | 1.2469% |
| V11 | 39 | 54 | 35.7136% | 83.4795% | 82.6998% | 2.2444% |
| V12 | 50 | 65 | 33.5425% | 79.6784% | 78.8499% | 1.2469% |

V12 réduit le CER de test de 1.3376 point par rapport à V10 : 130 erreurs de caractères en moins sur 9719 caractères. Le WER baisse de 0.0975 point, soit seulement 2 erreurs de mots en moins sur 2052 mots. Il reste élevé. Le nombre de segments exactement reconnus reste de 5 sur 401.

Le train de V12 est reconnu sans erreur, face à 41,97 % de CER de validation et 33,54 % de CER de test. L'écart de généralisation reste important. Les objectifs CER ≤ 20 % et WER ≤ 30 % ne sont pas atteints. Une seule graine ne permet pas de conclure à une supériorité générale ni à la significativité des écarts.

V10 : six convolutions, un canal gris, batch 24. V11 : six convolutions, gris + encre, batch 24. V12 : neuf convolutions, gris + encre, dropout CNN 0,05, batch 20 et organisation des poolings différente. L'effet de la seule profondeur n'est donc pas isolé. Les autres protocoles et anciens tests ne sont pas directement comparables à cette campagne.

Les scores de test sont rapportés pour documenter ces expériences terminées. Les réglages des expériences suivantes doivent être sélectionnés sur la validation. Les dénominateurs, compteurs et empreintes sont dans comparison_v9_v10_v11_v12.json. Les CER/WER/UWER de train et test ne sont pas disponibles à chaque époque : seul leur bilan final est mesuré.

[Comparaison détaillée](COMPARISON.md) · [Notes de publication et toutes les époques](RELEASE_NOTES.md) · [Vérifications](verification.json).
