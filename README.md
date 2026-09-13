# OCR AJAMI Hausa — V7

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
