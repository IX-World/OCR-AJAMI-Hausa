# V14 — essais A à D depuis V12 — IX-World

Les quatre essais affinent **indépendamment le même V12**, checkpoint `best_cer.pt`
de l'époque 50. Ils ne s'enchaînent pas par transfert de poids : B ne charge pas A,
C ne charge pas B et D ne charge pas C. Chaque essai crée un nouvel AdamW et un
nouveau scheduler. Le checkpoint V12 et ses fichiers sont conservés.
Le témoin A est un affinage contrôlé, pas une reproduction de l'entraînement
historique : le LR et les microbatches sont adaptés, de la même façon pour A–D.

| Essai | Augmentation du train | Optimisation | Perte |
|---|---|---|---|
| A | Huit vues fixes identiques à V12 | AdamW | CTC |
| B | Mêmes huit familles, paramètres renouvelés par époque | AdamW | CTC |
| C | Identique à B | SAM + AdamW | CTC |
| D | Identique à B | SAM + AdamW | CTC + MWER auxiliaire échantillonné |

Ce sont des expériences à mesurer, sans amélioration annoncée à l'avance.
Le préentraînement visuel auto-supervisé n'est pas inclus dans A–D.
V14 n'utilise pas AfriBERTa et n'exige aucune nouvelle bibliothèque.

## Données et poids figés

- Dataset local : `D:\ai_test\Hausa_repo_2026-10-15`.
- Dataset HF : `IntelligenceResearchLab/Hausa`, révision
  `1fb85fa070b6cf6432f1d45d21f78851d57f8944`.
- Protocole : `D:\ai_test\protocol_hausa_all_grouped_20261016`.
- **4 972 train / 200 validation / 401 test**, mêmes identifiants, références,
  groupes de provenance et vocabulaire que V9–V13. Aucun nouveau split.
- Split SHA-256 : `fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5`.
- Vocabulaire SHA-256 : `f92c368520aad95130f909a708fb58864e1f766f10f87d18c96690d1b20ced59`.
- Initialisation : `D:\ai_test\runs_hausa_all_grouped_20261016\v12_seed42\best_cer.pt`.
- SHA-256 V12 : `99a0d0f2c824a242b1b5e4864a5af2f4df0c291064ad0f1eba2ddeb6d5c3f6c1`.
- Les poids, le vocabulaire, le split et la recette CNN/augmentation V12 sont
  vérifiés avant un nouvel essai. Aucun checkpoint arbitraire n'est accepté.

Les sources sont séparées ; les sources 1 et 2, attribuées au même copiste connu,
sont regroupées. Les copistes non identifiés et les quasi-doublons restent une
limite du protocole. L'origine publique `train`/`test` d'un fichier désigne son
dossier sur disque ; son rôle expérimental est le champ `split`. Certaines images
de validation se trouvent donc légitimement dans le dossier public `data/test`.
Les références du test expérimental ne participent à aucune perte.

## Commandes CMD — une seule fenêtre d'entraînement à la fois

Les raccourcis sont installés dans `D:\ai_test`. Ils trouvent l'environnement
`.venv-hausa` ou l'environnement `.venv-v10` existant ; `HAUSA_PYTHON` permet
d'indiquer explicitement un autre interpréteur compatible PyTorch/Pillow.

Contrôle facultatif, sans entraînement :

```cmd
cd /d "D:\ai_test"
call verifier_v14.cmd --device cpu
```

Essai A :

```cmd
cd /d "D:\ai_test"
call lancer_v14_a.cmd
```

Après sa fin, essai B :

```cmd
cd /d "D:\ai_test"
call lancer_v14_b.cmd
```

Après sa fin, essai C :

```cmd
cd /d "D:\ai_test"
call lancer_v14_c.cmd
```

Après sa fin, essai D :

```cmd
cd /d "D:\ai_test"
call lancer_v14_d.cmd
```

Un dossier existant n'est jamais écrasé. Après une interruption, reprendre le même
essai, par exemple C, depuis sa dernière époque complète :

```cmd
cd /d "D:\ai_test"
call reprendre_v14.cmd --trial C
```

La reprise conserve son optimiseur, son scheduler, les graines et son budget.
Elle utilise automatiquement le code archivé du run, même après une mise à jour Git.
Un essai terminé ne peut pas être prolongé par cette commande : une nouvelle
expérience doit avoir un nouvel `--output` et être identifiée séparément.

## Réglages GTX 1060 6 Go

- Architecture V12 conservée : neuf convolutions, BatchNorm, dropout CNN 0,05,
  BiLSTM deux couches de 256 unités par direction, dropout 0,2, CTC 80 classes.
  **7 621 840 paramètres** avec le vocabulaire commun. Aucun empaquetage nouveau.
- Deux canaux : gris dans [-1,1] et encre locale, fond gaussien de rayon 5,
  gain 1,6 ; hauteur 96, largeur proportionnelle, Lanczos, miroir RTL, padding
  blanc pour le gris et sans encre pour le second canal.
- **Float32**, workers 0, microbatch jusqu'à **4**, accumulation de **5**
  microbatches : jusqu'à 20 vues par mise à jour. Un lot peut être plus petit.
- Budget de **600 000 éléments N×2×96×W**, donc les deux canaux et le padding
  sont comptés. Les lots sont regroupés par largeur ; aucune image n'est tronquée.
- Validation et bilan final avec le batch **20** de V12, dans l'ordre archivé.
  Cela conserve les conditions de padding du BiLSTM non empaqueté. Une modification
  de ce batch peut changer les prédictions : appliquer le même réglage aux quatre essais.
- AdamW : LR initial **0,0001**, weight decay **0,0001**, clipping de norme **5**.
- ReduceLROnPlateau sur CER validation, facteur 0,5, patience 3, minimum 0,000001.
- **30 époques maximum**, minimum 10, arrêt après 10 époques sans amélioration
  admissible du WER (seuil 0,0005). Graine 42, cuDNN benchmark, pas de déterminisme
  strict bit à bit. Ces budgets d'affinage sont communs aux quatre essais.
- Huit vues par image train : **39 776 vues par époque**, dans tous les essais.
  A réutilise les variantes V12 ; B/C/D changent leur graine à chaque époque.
  La vue originale reste présente. Toutes les images réelles du train sont utilisées.

Les familles sont : original, luminosité/contraste, variation d'épaisseur,
dégradation de résolution, flou, petite rotation, échelle géométrique et combinaison.
Le canal d'encre est calculé après transformation. Le code garde les paramètres
historiques pour isoler le renouvellement de la graine ; il ne démontre pas que
chaque dégradation préserve tous les points fins ou diacritiques Ajami.

Le pic réel de V14 reste à mesurer. SAM réalise deux passes successives et augmente
le temps de calcul. En cas d'erreur de mémoire, relancer les quatre essais dans de
nouveaux dossiers avec les mêmes réglages réduits, par exemple microbatch 2,
accumulation 10 et validation batch 10. Ne pas mélanger les deux recettes dans la
comparaison. Aucune nouvelle dépendance CUDA ni quantification n'est imposée.

## Calcul de SAM et MWER

SAM utilise une perturbation L2 de rayon **0,05**, calculée après l'accumulation
des gradients du groupe entier, avant clipping. La deuxième passe réutilise les
mêmes images et masques dropout ; les statistiques BatchNorm ne sont mises à jour
qu'à la première passe. Les paramètres originaux sont restaurés exactement avant
AdamW, y compris si une erreur interrompt la seconde passe.

Pour D : perte CTC + **0,05 × risque MWER** aux mises à jour échantillonnées.
Par défaut, une mise à jour sur quatre sélectionne uniformément **une vue TRAIN**
dans son groupe accumulé. Toutes les autres vues participent toujours à la CTC.
Il s'agit d'une approximation économique de l'objectif auxiliaire, pas d'un
apprentissage sur un sous-ensemble du corpus.

Le beam de largeur 8 conserve quatre hypothèses, avec huit labels non BLANK par
pas pour limiter le CPU. Le glouton et la référence TRAIN sont ajoutés, puis les
doublons retirés. Les probabilités de ces séquences sont recalculées par CTC,
de façon différentiable ; le risque est l'espérance de la distance d'édition sur
les mots divisée par le nombre de mots de la référence. Température 1.
La recherche des candidats est approchée ; le recalcul de leurs probabilités CTC
est exact. Les candidats sont réutilisés à la seconde passe SAM.
Les hyperparamètres et le nombre de vues effectivement soumises à MWER sont archivés.

MWER s'applique aux graphies strictes. Il ne remplace pas le WER par l'UWER et
n'efface pas les diacritiques des références. Le décodage d'évaluation reste
glouton CTC dans les quatre essais, afin d'isoler les changements d'apprentissage.

## Sélection et séparation du test

Avant l'affinage, chaque essai réévalue V12 sur les **200 images de validation**.
Ce score fixe son plafond CER. Le compromis retenu est le **WER de validation
minimal sous contrainte CER ≤ CER initial de V12** ; en cas d'égalité de WER,
le CER inférieur départage. Le checkpoint initial est candidat à l'époque 0 :
si aucun affinage n'est admissible, le script conserve V12 et indique zéro gain.

- `selected.pt` : ce compromis pour le bilan et le test éventuel.
- `best_cer.pt` : minimum CER sans autre contrainte.
- `best_wer.pt` : minimum WER sans contrainte CER, donc pas forcément admissible.
- `last.pt` : dernier état complet pour reprendre une interruption.

À la fin de chaque entraînement : CER, WER, UWER et exact sur **train et validation**,
une vue originale par image, avec `selected.pt`. Le test reste fermé.
Pour comparer les quatre essais uniquement sur la validation :

```cmd
cd /d "D:\ai_test"
call comparer_v14.cmd
```

La comparaison refuse des budgets, paramètres communs ou sources de code différents.
Elle ne lit pas les prédictions de test et ne choisit pas à partir des scores de test.
Un résultat préliminaire avec moins de quatre essais est signalé comme tel.

**Après le choix de la variante**, évaluer le test une seule fois. Exemple si C
est retenu (remplacer C par la lettre choisie) :

```cmd
cd /d "D:\ai_test"
call tester_v14.cmd --trial C
```

Cette commande ne réentraîne pas ; elle produit le bilan complet train/validation/test
et leurs prédictions avec le même `selected.pt`. Le V12 d'origine n'est plus requis
pour reprendre ou évaluer un V14 archivé. Le corpus et son protocole restent nécessaires.

## Résultats et attribution

Sorties distinctes :

```text
D:\ai_test\runs_hausa_v14_20261018\v14_A_seed42
D:\ai_test\runs_hausa_v14_20261018\v14_B_seed42
D:\ai_test\runs_hausa_v14_20261018\v14_C_seed42
D:\ai_test\runs_hausa_v14_20261018\v14_D_seed42
```

Chaque époque affiche CTC, MWER échantillonné, CER/WER/UWER de validation, exact,
LR, temps, VRAM allouée, pente/gain du CER, cinq références/prédictions et les
meilleurs scores. Les fichiers `history.csv`, `epochs.json`, `EPOCHS.md` conservent
toutes les mesures. Les CER/WER/UWER du train ne sont pas évalués à chaque époque.

`config.json`, `initial_validation.json`, `source/`, split, vocabulaire et protocole
assurent la traçabilité. `metrics.csv`/`metrics.json` et les `*_predictions.csv`
contiennent les bilans réels. Aucun score test n'est créé avant `--evaluate-test`.
CER/WER stricts gardent NFC, casse et diacritiques ; UWER applique casefold, NFD,
suppression des marques Mn puis NFC, en conservant ponctuation et lettres non décomposées.

Code, adaptation et entraînement : **IX-World**. Les crédits et droits des manuscrits
et annotations restent ceux du corpus. Une seule graine et les consultations
historiques du test limitent les conclusions de généralisation. Ces essais ne
garantissent pas CER ≤ 20 % ou WER ≤ 30 %. Les résultats pourront être publiés
après leur mesure ; aucun modèle V14 entraîné n'est annoncé ici.

Fondements : [SAM](https://arxiv.org/abs/2010.01412),
[MWER](https://arxiv.org/abs/1712.01818),
[HTR-VT](https://arxiv.org/abs/2409.08573).

## Contrôles du 18 octobre 2026

Les tests vérifient notamment la probabilité CTC et le gradient MWER contre une
énumération exhaustive, la restauration des paramètres SAM, BatchNorm, le dropout,
la pondération des microbatches inégaux, les quatre cycles synthétiques complets,
la reprise après interruption et l'absence d'évaluation du test par défaut.
Le prétraitement des huit vues de A reproduit celui de V12. Le contrôle réel
des poids et du split reproduit exactement les 20 premières prédictions V12
archivées sur CPU, sans apprentissage du corpus.

Un contrôle synthétique CUDA sur la GTX 1060 6 Go exécute C et D avec des tenseurs
4×2×96×640, accumulation 5 et les poids V12 chargés dans une copie de travail.
SAM et MWER produisent des pertes et gradients finis. Le maximum observé est
1 398 MiB alloués et 1 680 MiB réservés par PyTorch. Le détail est dans
[le contrôle GPU](configs/v14_gpu_smoke_20261018.json).
Ces valeurs ne sont ni le pic d'une époque complète ni toute la mémoire occupée
sur la carte ; elles ne permettent pas de comparer la consommation de C et D.
Aucune image du corpus n'a servi à ce contrôle CUDA, aucun poids n'a été sauvegardé
et aucun entraînement du corpus V14 n'a été lancé par ces vérifications.
