# OCR AJAMI Hausa — V10

`v10.py` contient la V10 sans commentaires. `Hausa_repo_nouveau/` contient le nouveau dataset local téléchargé depuis `IntelResearchLab/Hausa` sur Hugging Face, avec ses métadonnées et sa documentation d'origine.

Avec Python, Pillow et PyTorch installés, depuis la racine du dépôt :

```cmd
python v10.py --dataset ".\Hausa_repo_nouveau" --prepare-only
python v10.py --dataset ".\Hausa_repo_nouveau"
```

Sans argument `--dataset`, le script utilise `C:\ai-test\Hausa_repo_nouveau`.

Le découpage préparé contient 2519 lignes train, 280 validation et 400 test. Les huit vues fixes produisent 20152 exemples par epoch. Les résultats sont enregistrés dans un dossier daté sous `resultats_v10_nouveau`.

Les conditions d'utilisation et les crédits du dataset figurent dans [sa documentation](Hausa_repo_nouveau/README.md). Les images sont stockées directement dans Git, sans dépendance à Git LFS. Les caches de téléchargement, environnements Python et checkpoints d'entraînement ne font pas partie du dépôt.
