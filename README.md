# OCR AJAMI Hausa — V7

`v7_nouveau.py` remplace V10 comme script d'entraînement. Il utilise le réseau CNN + BiLSTM + CTC de V7, avec augmentation dynamique sur 55 % des lignes TRAIN. `Hausa_repo_nouveau/` contient le nouveau dataset local téléchargé depuis `IntelResearchLab/Hausa` sur Hugging Face, avec ses métadonnées et sa documentation d'origine.

Avec Python, Pillow et PyTorch installés, depuis la racine du dépôt :

```cmd
python v7_nouveau.py --dataset ".\Hausa_repo_nouveau" --train-count 3000 --val-count 100 --prepare-only
python v7_nouveau.py --dataset ".\Hausa_repo_nouveau" --train-count 3000 --val-count 100 --epochs 100
```

Sans argument `--dataset`, le script utilise `C:\ai-test\Hausa_repo_nouveau`.

Le découpage préparé contient 3000 lignes TRAIN, 100 validation et 99 TEST. Chaque epoch parcourt les 3000 lignes une fois, soit 125 batches de 24. Les augmentations changent selon l'epoch. Le maximum est de 100 epochs, avec l'arrêt adaptatif de V7. Les résultats sont enregistrés dans un dossier daté sous `resultats_v7_nouveau`.

Le meilleur checkpoint est choisi sur le CER de validation ; le TEST s'exécute automatiquement après l'entraînement. Les ensembles d'origine sont redistribués pour ce découpage : les scores ne sont pas directement comparables à ceux du précédent TEST de 299 lignes. Le vocabulaire est reconstruit sur le TRAIN uniquement.

Les conditions d'utilisation et les crédits du dataset figurent dans [sa documentation](Hausa_repo_nouveau/README.md). Les images sont stockées directement dans Git, sans dépendance à Git LFS. Les caches de téléchargement, environnements Python et checkpoints d'entraînement ne font pas partie du dépôt.
