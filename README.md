# OCR AJAMI Hausa — IX-World

## Campagne active — manuscrits séparés — 16 octobre 2026

Voir [le guide actuel](README_HAUSA_GROUPED_20261016.md) : **2 609 train /
229 validation / 400 tests publics**, sans manuscrit partagé. Les groupes de
copistes documentés sont également séparés. Les 2 335 exemples du train public
liés aux groupes du test sont exclus ; les deux annotations contradictoires
restent en quarantaine. Les dernières données HF sont bien présentes : 5 575 PNG.
Les quatre commandes `lancer_v9.cmd` à `lancer_v12.cmd` et les notebooks pointent
sur ce nouveau protocole. Nouveaux entraînements depuis zéro, sans anciens poids.
Publication des nouveaux modèles après mesure des résultats ; les poids initiaux
ne sont pas promus en production sous cette nouvelle définition du split.

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
**V13 reste une proposition, sans modèle implémenté ni résultat annoncé.**

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
