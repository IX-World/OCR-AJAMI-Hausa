# IX-World — tout le corpus Hausa, partitions séparées

## Données et effectifs

Le dataset `IntelligenceResearchLab/Hausa` a été téléchargé dans
`D:\ai_test\Hausa_repo_2026-10-15`. La vérification HF du 16 octobre 2026 confirme
la révision actuelle `1fb85fa070b6cf6432f1d45d21f78851d57f8944` : **5 575 PNG**.
Les corrections de chemins et de doublons sont conservées. Les deux images
annotées « da »/« mutane » restent en quarantaine sur décision de l’utilisateur.
**Les 5 573 autres images sont toutes utilisées. Aucune exclusion de provenance
n’est nécessaire dans cette nouvelle répartition.**

| Partition | Images | Sources entières |
|---|---:|---|
| Train | 4 972 | 1, 2, 4, 8, 9, 10, 11, 16, 17 |
| Validation | 200 | 7, 15 |
| Test | 401 | 3, 6, 18 |
| Quarantaine des annotations contradictoires | 2 | hors expérience |
| Total HF téléchargé | 5 575 | |

La cible indicative 4 800/200/400 ne consommait que 5 400 images. Les effectifs
retenus utilisent tout le corpus valide : des groupes entiers sont sélectionnés
au plus près de 200 exemples de validation, puis de 400 exemples de test.
Le test compte 401 images afin de conserver les manuscrits entiers. Les décisions
reposent sur la provenance et les effectifs, sans consulter les scores.

Le test est **reconstruit**, conformément au choix de l’utilisateur : il diffère
du test public initial de 400 images. Les images, transcriptions et sources HF
restent inchangées. Le champ `partition` garde le dossier public d’origine
(`train` ou `test`) pour retrouver chaque fichier ; le champ `split` indique
son appartenance expérimentale. Dans les prédictions, `evaluated_partition`
indique également l’ensemble réellement évalué.

## Indépendance

Aucun identifiant d’image, hash de pixels exact, manuscrit ou groupe de copiste
identifié n’est partagé entre les trois ensembles. Les sources 1 et 2 ont le
même copiste documenté, Ibrahim Yaqubu Gusau, et restent ensemble dans le train.
Les identités des autres copistes ne sont pas documentées : une indépendance
entre tous les scripteurs inconnus ne peut pas être certifiée. Les personnes qui
annotent ou relisent ne sont pas assimilées aux auteurs des écritures originales.
La recherche de quasi-doublons visuels reste distincte du contrôle des pixels exacts.
Les mots usuels peuvent réapparaître dans des manuscrits différents.

Les augmentations sont créées après la séparation, uniquement pour le train.
Le vocabulaire est construit sur le train seul ; les références inconnues restent
inchangées dans les scores. Les manifestes et empreintes sont conservés dans
`D:\ai_test\protocol_hausa_all_grouped_20261016`.

## Commandes CMD

Dataset commun : `D:\ai_test\Hausa_repo_2026-10-15`.
Protocole commun : `D:\ai_test\protocol_hausa_all_grouped_20261016`.
Sorties : `D:\ai_test\runs_hausa_all_grouped_20261016\VERSION_seed42`.

Exécuter un modèle à la fois :

```cmd
cd /d "D:\ai_test"
call lancer_v9.cmd
```

Puis `lancer_v10.cmd`, `lancer_v11.cmd`, `lancer_v12.cmd`. Pour les enchaîner,
utiliser `run_v9_v12.cmd`. Les quatre notebooks dédiés dans
`D:\ai_test\OCR-AJAMI-Hausa\notebooks` pointent sur le même protocole.
Les raccourcis imposent `--fresh-start` et `--require-group-disjoint`.
Les anciens runs ne sont pas écrasés et leurs checkpoints restent archivés.

| Modèle | Augmentation | Vues/époque | Batch | Batches/époque |
|---|---|---:|---:|---:|
| V9 | Dynamique à 30 % | 4 972 | 24 | 208 |
| V10 | Original + 7 vues fixes | 39 776 | 24 | 1 658 |
| V11 | 8 vues, gris + encre | 39 776 | 24 | 1 658 |
| V12 | 8 vues, encre, CNN profond | 39 776 | 20 | 1 989 |

À chaque époque : loss train, CER/WER/UWER et taux exact de validation,
pente et gain du CER, LR, temps, VRAM, structure, cinq exemples REF/PRED,
meilleurs scores et patience. Plafond 100 époques, minimum 30, patience 15 ;
AdamW à 0,001 et ReduceLROnPlateau sur le CER de validation.
Le même checkpoint, sélectionné sur ce CER, produit le bilan final des trois
partitions sur les images originales sans augmentation. Résultats et compteurs :
`metrics.csv`, `metrics.json`, les trois `*_metrics.json`, prédictions et README.
L’UWER normalise la casse et les diacritiques ; le WER strict est conservé.

Pour recalculer le bilan après entraînement : `tester_v9.cmd`, ou la version
concernée. Aucun apprentissage n’est effectué par cette commande.

## Initialisation et conservation des poids

« Depuis zéro » signifie **initialisation aléatoire**, jamais tous les poids
numériquement nuls. Les anciens modèles avaient déjà appris certaines images
qui appartiennent au nouveau test. Ils ne peuvent pas initialiser cette nouvelle
comparaison indépendante. Leurs fichiers sont conservés pour l’historique.

Après un entraînement sur ce nouveau split, on peut reprendre son propre run,
avec le même split et le même vocabulaire, sans réinitialiser ses poids :

```cmd
cd /d "D:\ai_test\OCR-AJAMI-Hausa"
call python_ocr.cmd --version v9 --require-group-disjoint --resume
```

Cette commande exige `last.pt` dans le nouveau dossier du modèle. Les raccourcis
`lancer_*` restent réservés à un nouveau départ et refusent `--resume`.

## Reconstituer et publier

Après téléchargement et corrections initiales avec `prepare_hausa.py` :

```cmd
cd /d "D:\ai_test\OCR-AJAMI-Hausa"
"C:\Users\admin\Documents\ChatGPT\AJAMI\.venv-v10\Scripts\python.exe" prepare_hausa_grouped.py --dataset "D:\ai_test\Hausa_repo_2026-10-15" --base-protocol "D:\ai_test\protocol_hausa_20261015" --output "D:\ai_test\protocol_hausa_all_grouped_20261016" --all-images
```

Le scénario antérieur 2 609/229/400, qui excluait 2 335 images pour préserver
exactement le test public, est archivé et n’est plus utilisé par les commandes.
La V9 initiale 4 656/517/400 et la V10 arrêtée restent également archivées.
Leurs scores ne décrivent pas ce nouveau protocole.

Aucun modèle de cette nouvelle campagne n’est encore entraîné ni publié sur HF.
Les nouveaux checkpoints, leurs résultats réels, le split et un README précis
seront packagés après entraînement. Les scripts HF/GitHub refusent désormais un
protocole qui partage les manuscrits ou qui écarte une partie du corpus valide.
Le dépôt V9 prévu reste `IntelligenceResearchLab/Hausa-OCR-AJAMI-v9-20261015` ;
le code et les commits GitHub sont attribués à **IX-World**. V13 reste une proposition.
