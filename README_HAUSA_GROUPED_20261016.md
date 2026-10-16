# IX-World — campagne avec séparation des manuscrits

Cette campagne remplace le protocole initial 4 656/517/400. Les nouvelles V9,
V10, V11 et V12 démarrent avec des poids aléatoires. Aucun checkpoint antérieur
n'initialise leur entraînement. Les architectures et augmentations sont conservées.

## Données effectivement téléchargées

Dataset : `IntelligenceResearchLab/Hausa`, révision
`1fb85fa070b6cf6432f1d45d21f78851d57f8944`.
Une vérification directe de Hugging Face le 16 octobre 2026 confirme que cette
révision est toujours la dernière : **5 575 PNG** (5 175 train public et 400 test).
Les deux images aux références contradictoires « da »/« mutane » restent en
quarantaine, soit 5 573 exemples utilisables avant séparation par provenance.
Les corrections de chemins et de doublons précédemment vérifiées sont conservées.

## Séparation commune aux quatre modèles

| Ensemble | Images réelles | Sources |
|---|---:|---|
| Train | 2 609 | 8, 9, 17 |
| Validation | 229 | 6, 7, 10, 18 |
| Test public | 400 | 1, 3, 4, 11, 15, 16 |
| Exclusions liées aux groupes du test | 2 335 | 1, 2, 4, 15, 16 |
| Quarantaine des annotations contradictoires | 2 | conservées hors expérience |
| Total des images du dépôt | 5 575 | |

Aucun identifiant d'image, hash de pixels exact, manuscrit ou groupe de
copiste identifié n'est partagé entre train, validation et test.
Les sources 1 et 2 partagent le copiste Ibrahim Yaqubu Gusau et sont regroupées.
Les autres identités de copistes ne sont pas documentées : la séparation entre
tous les scripteurs inconnus ne peut pas être certifiée. Les personnes qui
annotent ou relisent le corpus ne sont pas assimilées aux copistes des manuscrits.
La recherche de quasi-doublons visuels reste distincte du contrôle des pixels exacts.

Le test public conserve ses 400 images et références ; ses métadonnées restent
identiques. Les exemples apparentés du train public sont exclus, sans effacer les
images. `excluded_public_train.json` conserve leurs identifiants et motifs.
Le vocabulaire est calculé uniquement sur les 2 609 références de train.
Les références inconnues ne sont pas réécrites pour réduire les erreurs.

La cible de validation est de 10 % des 2 838 exemples éligibles. Une recherche
déterministe sélectionne des groupes entiers en minimisant l'écart d'effectif,
sans consulter les prédictions : le résultat est 229 images, soit 8,07 %.
Les mots usuels peuvent exister dans plusieurs manuscrits ; la séparation porte
sur les images et leur provenance, et ne constitue pas une évaluation de mots
exclus du vocabulaire d'entraînement.

## Chemins locaux et commandes CMD

- Dataset : `D:\ai_test\Hausa_repo_2026-10-15`.
- Protocole : `D:\ai_test\protocol_hausa_grouped_20261016`.
- Résultats : `D:\ai_test\runs_hausa_grouped_20261016\VERSION_seed42`.
- Notebooks : `D:\ai_test\OCR-AJAMI-Hausa\notebooks` ; les quatre notebooks
  `v9_hausa_20261015.ipynb` à `v12_hausa_20261015.ipynb` pointent sur ce protocole.

Dans CMD, exécuter une seule commande de lancement à la fois :

```cmd
cd /d "D:\ai_test"
call lancer_v9.cmd
```

Puis, lorsque le modèle précédent est terminé : `lancer_v10.cmd`,
`lancer_v11.cmd`, `lancer_v12.cmd`. `run_v9_v12.cmd` les enchaîne dans cet ordre.
Les commandes imposent `--fresh-start` et `--require-group-disjoint`.
Un ancien protocole ou checkpoint est refusé. Aucun ancien run n'est écrasé.

| Modèle | Vues train par époque | Batch par défaut | Batches par époque |
|---|---:|---:|---:|
| V9 | 2 609 ; augmentation dynamique à 30 % | 24 | 109 |
| V10 | 20 872 ; huit vues fixes | 24 | 870 |
| V11 | 20 872 ; huit vues fixes, canal d'encre | 24 | 870 |
| V12 | 20 872 ; huit vues fixes, canal d'encre, CNN profond | 20 | 1 044 |

Les augmentations sont créées après la séparation et uniquement pour le train.
Chaque époque affiche la loss, CER/WER/UWER et taux exact de validation, pente et
gain du CER, LR, temps, VRAM, structure, exemples REF/PRED et meilleurs scores.
Le checkpoint est sélectionné sur le CER de validation. À la fin, le même modèle
est évalué sur les trois ensembles avec une vue originale par image.
`metrics.csv`, `metrics.json`, les trois `*_metrics.json`, les prédictions et le
README contiennent CER, WER, UWER et taux exact. L'UWER normalise la casse et
les diacritiques, en conservant ponctuation et lettres haoussa non décomposables.

Pour recalculer le bilan après un entraînement terminé :

```cmd
cd /d "D:\ai_test"
call tester_v9.cmd
```

Remplacer `v9` par la version concernée. Les règles d'optimisation et d'arrêt
restent harmonisées : plafond 100 époques, minimum 30, patience 15, AdamW à 0,001,
ReduceLROnPlateau sur le CER de validation. V13 reste une proposition.

## Reconstituer le protocole

Après le téléchargement et les corrections initiales `prepare_hausa.py` :

```cmd
cd /d "D:\ai_test\OCR-AJAMI-Hausa"
"C:\Users\admin\Documents\ChatGPT\AJAMI\.venv-v10\Scripts\python.exe" prepare_hausa_grouped.py --dataset "D:\ai_test\Hausa_repo_2026-10-15" --base-protocol "D:\ai_test\protocol_hausa_20261015" --output "D:\ai_test\protocol_hausa_grouped_20261016"
```

Le programme vérifie les images et les métadonnées, conserve l'ancien protocole
et refuse d'écraser un split différent. Les manifestes individuels `train.jsonl`,
`validation.jsonl`, `test.jsonl`, les groupes, exclusions et empreintes sont archivés.

## Résultats initiaux et publication

La V9 initiale reste dans `runs_hausa_20261016_zero\v9_seed42`. Son bilan est
CER/WER/UWER : train 0,28/1,17/1,14 %, validation 21,60/57,58/54,55 %,
test 27,07/77,84/76,81 %. Le recalcul CPU reproduit exactement les 400 prédictions
initialement obtenues sur GPU. Ces résultats ne décrivent pas le nouveau split.
Ils montrent un ajustement très fort au train et une généralisation limitée.
La V10 initiale a été arrêtée sur demande ; ses checkpoints de l'époque 1 restent
dans son ancien dossier. Aucun de ces fichiers n'initialise la nouvelle campagne.

La mise en production des anciens poids est en attente. Les scripts de packaging
et de publication HF/GitHub refusent désormais un protocole qui partage des groupes.
Après de nouveaux entraînements, chaque modèle aura son README et son package.
Le dépôt prévu pour V9 reste `IntelligenceResearchLab/Hausa-OCR-AJAMI-v9-20261015`,
avec contribution GitHub **IX-World**. Aucun nouveau score ni modèle de ce protocole
n'est annoncé avant sa mesure effective.
