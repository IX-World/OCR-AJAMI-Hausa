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

La **V9 est entraînée**, avec un checkpoint sélectionné à l'époque 33 sur
48 époques exécutées. Test : CER **40,6935 %**, WER **86,9396 %**, UWER **86,1598 %**.
Les résultats train/validation/test et l'audit sont documentés dans
[son README](models/v9_all_grouped_20261016/README.md). L'écart important avec le
train limite la généralisation ; ce n'est pas une revendication de transcription fiable.
Les [48 époques mesurées](models/v9_all_grouped_20261016/EPOCHS.md) sont exposées
en tableaux et courbes PNG/SVG, ainsi qu'en CSV et JSON pour comparaison.
Le packaging génère désormais les tableaux et le JSON de chaque modèle à partir
de son historique réel ; les courbes sont optionnelles et nécessitent Matplotlib.
Les poids sont publiés sur
[Hugging Face](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v9-20261015)
et dans une [release GitHub](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v9-hausa-all-grouped-20261016).
Le nom du dépôt HF conserve la date initialement prévue, tandis que la fiche
documente l'entraînement réel du 16 octobre et le split de 5 573 images valides.
La **V10 est également entraînée** : 54 époques, checkpoint CER retenu à l'époque 39.
Test : CER **34,8801 %**, WER **79,7758 %**, UWER **79,2398 %**.
Le [README V10](models/v10_all_grouped_20261016/README.md),
les [valeurs des 54 époques](models/v10_all_grouped_20261016/EPOCHS.md)
et le [diagnostic du WER](models/v10_all_grouped_20261016/WER_ANALYSIS.md)
documentent aussi le fort écart train/test. Les poids sont disponibles sur
[HF V10](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v10-20261015)
et dans la [release V10](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v10-hausa-all-grouped-20261016).
La **V11 est publiée le 17 octobre 2026** : 54 époques, checkpoint CER à l'époque 39.
Test : CER **35,7136 %**, WER **83,4795 %**, UWER **82,6998 %**. V11 reprend V10
avec un canal d'encre locale, mais ne l'améliore pas sur ce test pour cette graine.
Le train entièrement reconnu et les scores de validation/test indiquent un fort
écart de généralisation. Voir le [README V11](models/v11_all_grouped_20261017/README.md),
les [54 époques](models/v11_all_grouped_20261017/EPOCHS.md),
le [modèle HF](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v11-20261015)
et la [release GitHub](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v11-hausa-all-grouped-20261017).
Les autres modèles seront publiés après leurs entraînements et évaluations réels.
Les scripts HF/GitHub refusent un protocole partageant les manuscrits ou écartant
une partie du corpus valide. Le code et les commits GitHub sont attribués à
**IX-World**. V13 reste une proposition.
