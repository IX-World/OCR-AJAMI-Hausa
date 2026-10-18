# OCR AJAMI Hausa — IX-World

## Campagne active — tout le corpus et manuscrits séparés — 16 octobre 2026

Voir [le guide actuel](README_HAUSA_GROUPED_20261016.md) : **4 972 train /
200 validation / 401 test**, soit les **5 573 images valides** du dépôt HF.
Seules les deux annotations contradictoires restent en quarantaine (5 575 PNG au total).
Le test est reconstruit par manuscrit ; les partitions publiques initiales sont
conservées sur disque et tracées comme provenance. Aucun manuscrit n’est partagé.
Les sources 1 et 2 sont regroupées par copiste documenté ; les autres identités
restent inconnues. Commandes et notebooks utilisent ce protocole commun.
Les anciens checkpoints sont archivés ; les nouveaux modèles sont initialisés
aléatoirement pour éviter l’exposition préalable aux nouvelles données de test.

### V9 entraînée — 16 octobre 2026

Le checkpoint de l'époque **33** est disponible sur
[Hugging Face](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v9-20261015)
et dans la [release GitHub avec les poids](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v9-hausa-all-grouped-20261016).
Les [résultats et le README détaillé](models/v9_all_grouped_20261016/README.md)
documentent le split **4 972 / 200 / 401**, l'entraînement depuis une initialisation
aléatoire et l'audit sans chevauchement d'images exactes ni de manuscrits.
Les [valeurs des 48 époques et leurs courbes](models/v9_all_grouped_20261016/EPOCHS.md)
sont publiées en tableaux, [CSV brut](models/v9_all_grouped_20261016/history.csv)
et [JSON numérique](models/v9_all_grouped_20261016/epochs.json), avec les unités.

| Partition | CER | WER | UWER |
|---|---:|---:|---:|
| Train | 4,60 % | 17,10 % | 16,30 % |
| Validation | 47,28 % | 89,93 % | 87,92 % |
| Test | 40,69 % | 86,94 % | 86,16 % |

La publication expose un résultat expérimental reproductible. L'écart entre
train et test montre une forte limite de généralisation. Ces scores ne se
comparent pas directement à ceux des anciens tests. Les identités des copistes
non documentés et les quasi-doublons restent à vérifier.

### V10 entraînée — 16 octobre 2026

Le checkpoint de l'époque **39**, retenu après **54 époques**, est publié sur
[Hugging Face](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v10-20261015)
et dans la [release GitHub](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v10-hausa-all-grouped-20261016).
Voir le [README complet](models/v10_all_grouped_20261016/README.md),
les [54 époques et leurs courbes](models/v10_all_grouped_20261016/EPOCHS.md)
et le [diagnostic du WER et les propositions](models/v10_all_grouped_20261016/WER_ANALYSIS.md).

| Partition | CER | WER | UWER |
|---|---:|---:|---:|
| Train | 0,0021 % | 0,0094 % | 0,0094 % |
| Validation | 43,56 % | 88,14 % | 86,80 % |
| Test | 34,88 % | 79,78 % | 79,24 % |

Le CER de test diminue de 5,81 points par rapport à V9 sur le même split.
Le train presque parfaitement reconnu et le WER de test élevé montrent des signes
forts de surapprentissage et une généralisation insuffisante. Le modèle est publié
comme résultat expérimental reproductible ; sa qualité ne permet pas de revendiquer
une transcription automatique fiable.

V10 et V11 partagent le même CNN à six convolutions, BiLSTM–CTC, huit vues,
batch 24 et split 4 972/200/401. V11 ajoute un canal d'encre locale à l'image
en niveaux de gris : fond gaussien de rayon 5 pixels, différence positive fond–gris,
gain 1,6, puis normalisation. Les deux apprentissages partent d'une initialisation
aléatoire ; les résultats V11 sont rapportés ci-dessous. Le code V13 est documenté
dans [son guide distinct](README_V13.md), avec ses résultats du 18 octobre ci-dessous.

### V11 entraînée — publication du 17 octobre 2026

Le checkpoint de l'époque **39**, sélectionné après **54 époques**, est publié sur
[Hugging Face](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v11-20261015)
et dans la [release GitHub](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v11-hausa-all-grouped-20261017).
Voir le [README V11](models/v11_all_grouped_20261017/README.md),
les [54 époques et les courbes](models/v11_all_grouped_20261017/EPOCHS.md)
et la [comparaison détaillée](models/v11_all_grouped_20261017/RELEASE_NOTES.md).

| Partition | CER | WER | UWER |
|---|---:|---:|---:|
| Train | 0,00 % | 0,00 % | 0,00 % |
| Validation | 41,70 % | 86,80 % | 83,45 % |
| Test | 35,71 % | 83,48 % | 82,70 % |

L'ajout du canal d'encre ne démontre pas de gain sur ce test : V10 mesurait
CER 34,88 % et WER 79,78 %. Le train est entièrement reconnu, mais la généralisation
reste insuffisante. Cette comparaison porte sur une seule graine et ne constitue
pas une conclusion générale sur les cartes d'encre. Les objectifs CER ≤ 20 % et
WER ≤ 30 % ne sont pas atteints.

### V12 entraînée — publication du 18 octobre 2026

Le checkpoint de l'époque **50**, sélectionné après **65 époques**, est publié sur
[Hugging Face](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v12-20261015)
et dans la [release GitHub](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v12-hausa-all-grouped-20261018).
Voir le [README V12](models/v12_all_grouped_20261018/README.md),
les [65 époques et leurs courbes](models/v12_all_grouped_20261018/EPOCHS.md)
et la [comparaison V9–V12](models/v12_all_grouped_20261018/COMPARISON.md).

| Partition | CER | WER | UWER |
|---|---:|---:|---:|
| Train | 0,00 % | 0,00 % | 0,00 % |
| Validation | 41,97 % | 84,12 % | 81,43 % |
| Test | 33,54 % | 79,68 % | 78,85 % |

V12 utilise neuf convolutions, deux canaux gris/encre et 7 621 840 paramètres.
Sur ce test, le CER baisse de 1,3376 point par rapport à V10 ; le WER baisse
de seulement 0,0975 point, soit deux erreurs de mots en moins sur 2 052 mots.
Le train sans erreur et le test encore difficile montrent un fort écart de
généralisation. Les changements de profondeur, pooling, dropout et batch ne
constituent pas une ablation isolant l'effet des couches supplémentaires.

### Beam search V10 sur validation uniquement

Le [script et sa commande CMD](README_BEAM_V10_VALIDATION.md) comparent le
glouton et le beam search sur les **200 images de validation**, sans réentraîner
V10 ni évaluer le test expérimental. Le glouton doit reproduire exactement les
prédictions historiques. Les émissions sont mises en cache pour les essais suivants.

```cmd
cd /d "D:\ai_test"
call tester_beam_v10_validation.cmd
```

Le premier essai CPU du 17 octobre donne un WER de validation de 88,1432 % pour
le glouton et 88,3669 % pour les beams 10 et 25, sans modèle de langue. Il ne
montre donc pas d'amélioration du WER. Les hypothèses n-best sont enregistrées
pour préparer un futur reclassement linguistique ; aucun apprentissage sémantique
n'est effectué dans ce premier essai.

## V13 : supervision sémantique contextuelle

Entraînement terminé le **18 octobre 2026**, **49 époques**, checkpoint **34**.
Les poids et le README complet sont sur
[HF](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v13-20261015)
et dans la [release GitHub](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v13-hausa-all-grouped-20261018).
Voir la [comparaison V9–V13](models/v13_all_grouped_20261018/COMPARISON.md)
et les [49 époques, pertes séparées et courbes](models/v13_all_grouped_20261018/EPOCHS.md).

| Partition | CER | WER | UWER |
|---|---:|---:|---:|
| Train | 0,0021 % | 0,0094 % | 0,0094 % |
| Validation | 44,37 % | 93,29 % | 91,50 % |
| Test | 39,10 % | 86,31 % | 86,01 % |

**V13 régresse face à V10/V12 et le surapprentissage persiste.** Il est parti
de poids OCR aléatoires. Les branches époque/localité sont restées masquées,
faute de métadonnées ; aucune compréhension historique/régionale n'est démontrée.
Les changements d'architecture et de recette ne permettent pas d'isoler l'effet
de la supervision sémantique sans une ablation supplémentaire.

Le [code V13 et son guide CMD](README_V13.md) conservent le split commun
4 972/200/401 et ajoutent une supervision auxiliaire AfriBERTa figée, préparée
sur CPU à partir du train uniquement. Le CNN–BiLSTM–CTC est initialisé de zéro ;
les microbatches, l'accumulation et le budget de pixels visent la GTX 1060 6 Go.
Le contexte documenté de langue, d'époque et de localité conditionne le réseau.
Les époques/localités absentes restent inconnues et masquées ; aucune capacité
historique ou régionale ni aucun gain n'est annoncé sans annotations et résultats.

```cmd
cd /d "D:\ai_test"
call preparer_v13.cmd && call lancer_v13.cmd
```

## Protocole initial archivé — 15 octobre 2026

Les commandes locales, corrections du dataset, configurations et procédures
de publication sont détaillées dans [README_HAUSA_20261015.md](README_HAUSA_20261015.md).
Le split commun conserve les **400 exemples du test public** à part ; les
5 173 exemples publics de train nettoyés donnent **4 656 train / 517 validation**.
Les architectures et augmentations sont documentées, les résultats et README
de chaque modèle seront générés uniquement après les entraînements locaux.
La campagne active du 16 octobre repart de zéro pour les quatre versions,
avec une initialisation aléatoire et les sorties dans
`D:\ai_test\runs_hausa_20261016_zero`. Les commandes de lancement imposent
`--fresh-start` et refusent de charger un ancien checkpoint pour initialiser le train.
**V13 dispose d'un [code séparé](README_V13.md) et de résultats du 18 octobre 2026
sur le protocole actif 4 972/200/401 présenté plus haut.**

Les nouveaux lancements affichent un bilan détaillé à chaque époque, avec cinq
exemples de validation. Le bilan final exporte CER, WER, UWER (casse et diacritiques
normalisés) et taux exact sur train/validation/test dans `metrics.csv` et
`metrics.json`. Pour compléter un run déjà lancé avec l'ancien script, exécuter
`tester_v9.cmd` (ou sa version V10/V11/V12) après sa fin, sans réentraîner.

Les éléments ci-dessous documentent les expériences historiques.

## Modèle V10 entraîné et publication Hugging Face

Le dernier checkpoint **V10 nouveau 92/3/5**, issu de `train_hausa_ctc_v10_nouveau_92_3_5.py`, est publié sur [IntelligenceResearchLab/Hausa-OCR-AJAMI](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI). Sa copie, son vocabulaire, le découpage exact, ses résultats et le code d'inférence sont dans `models/v10/`.

Le checkpoint retenu est celui de l'epoch **25**, entraîné avec **2942 lignes TRAIN / 96 validation / 160 TEST**. La vérification du TEST donne **CER 25,496 %**, **WER 69,631 %** et **27,5 % de lignes exactes**. Ce test diffère de celui de l'ancienne V10 ; les scores ne sont pas directement comparables. L'ancienne version reste accessible dans l'historique Git.

Cette publication héberge les fichiers du modèle ; elle ne crée pas un serveur d'inférence.

## Entraînement V7

`v7_nouveau.py` remplace V10 comme script d'entraînement. Il utilise le réseau CNN + BiLSTM + CTC de V7, avec augmentation dynamique sur 55 % des lignes TRAIN. `Hausa_repo_nouveau/` contient le nouveau dataset local téléchargé depuis `IntelligenceResearchLab/Hausa` sur Hugging Face, avec ses métadonnées et sa documentation d'origine.

Avec Python, Pillow et PyTorch installés, depuis la racine du dépôt :

```cmd
python v7_nouveau.py --dataset ".\Hausa_repo_nouveau" --train-count 3000 --val-count 100 --prepare-only
python v7_nouveau.py --dataset ".\Hausa_repo_nouveau" --train-count 3000 --val-count 100 --epochs 100
```

Sans argument `--dataset`, le script utilise `C:\ai-test\Hausa_repo_nouveau`.

Le découpage préparé contient 3000 lignes TRAIN, 100 validation et 99 TEST. Chaque epoch parcourt les 3000 lignes une fois, soit 125 batches de 24. Les augmentations changent selon l'epoch. Le maximum est de 100 epochs, avec l'arrêt adaptatif de V7. Les résultats sont enregistrés dans un dossier daté sous `resultats_v7_nouveau`.

Le meilleur checkpoint est choisi sur le CER de validation ; le TEST s'exécute automatiquement après l'entraînement. Les ensembles d'origine sont redistribués pour ce découpage : les scores ne sont pas directement comparables à ceux du précédent TEST de 299 lignes. Le vocabulaire est reconstruit sur le TRAIN uniquement.

Les conditions d'utilisation et les crédits du dataset figurent dans [sa documentation](Hausa_repo_nouveau/README.md). Les images sont stockées directement dans Git, sans dépendance à Git LFS. Les caches de téléchargement et environnements Python ne font pas partie du dépôt. Le checkpoint V10 publié est conservé dans models/v10/.
