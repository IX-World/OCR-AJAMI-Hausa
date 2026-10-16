# V9 — corpus Hausa complet et manuscrits séparés

Contribution et entraînement : **IX-World**. Publication du modèle entraîné
localement le 16 octobre 2026 sur une NVIDIA GeForce GTX 1060 6 GB.

Le checkpoint publié est celui de l'époque **33**, sélectionné uniquement sur
le minimum du CER de validation, après **48 époques exécutées** et arrêt anticipé.
Initialisation aléatoire ; aucun checkpoint antérieur utilisé.

| Partition | Segments | CER | WER | UWER | Segments exacts |
|---|---:|---:|---:|---:|---:|
| Train | 4 972 | 4,5971 % | 17,0967 % | 16,2991 % | 84,1714 % |
| Validation | 200 | 47,2777 % | 89,9329 % | 87,9195 % | 18,0000 % |
| Test | 401 | 40,6935 % | 86,9396 % | 86,1598 % | 1,2469 % |

L'UWER est ici le WER après normalisation Unicode de la casse et des diacritiques.
Les trois ensembles sont évalués avec le même checkpoint, sans augmentation.
Le taux exact de test correspond à **5 segments sur 401**. Ces résultats
documentent une forte limite de généralisation : la publication est une référence
expérimentale reproductible, sans revendication de transcription autonome fiable.
Les scores ne sont pas directement comparables à ceux des anciens tests V9/V10.

## Données et indépendance

- Dataset : `IntelligenceResearchLab/Hausa`, révision
  `1fb85fa070b6cf6432f1d45d21f78851d57f8944`.
- 5 573 images valides utilisées sur 5 575 PNG ; les deux annotations
  contradictoires restent en quarantaine, selon la décision de l'utilisateur.
- Train : sources 1, 2, 4, 8, 9, 10, 11, 16 et 17.
- Validation : sources 7 et 15 ; test : sources 3, 6 et 18.
- Aucun identifiant d'image, pixel exact, manuscrit ou groupe de copiste connu
  partagé. Les sources 1 et 2 sont regroupées dans le train.
- L'identité des autres copistes et l'absence de quasi-doublons restent à vérifier.
- Le test est reconstruit par groupes : ce n'est pas le test public initial de 400 images.

## Fichiers et vérifications

L'archive `hausa-v9-all-grouped-20261016.zip` contient `best_cer.pt`, le README
détaillé, les sources exactes d'entraînement et d'évaluation, le vocabulaire,
les partitions, la configuration, l'historique, les prédictions des trois
ensembles, leurs scores et l'audit. `manifest.json` donne les empreintes SHA-256
de tous les fichiers de cette archive complète ; les poids sont distribués dans
la release GitHub et sur Hugging Face, tandis que Git conserve les résultats textuels.
Aucune image du corpus n'est incluse dans ce package de modèle.

- Code d'entraînement : `c24c44b1c4d13e0e678782e63ce0bcefd445703f`.
- SHA-256 du checkpoint :
  `7095927d5cbfd89d8f26193d3bca9d5e57182f5954c0dc706bc7b5870b8f2e4f`.
- SHA-256 du split :
  `fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5`.
- CER/WER/UWER recalculés à partir des références et prédictions ; chargement
  strict des poids et passage avant CPU vérifiés ; 17 tests logiciels réussis.
- Durée enregistrée : **59,07 min** ; pic VRAM allouée : **5 037,6 MiB**.

Le modèle CNN–BiLSTM–CTC produit directement du texte latin depuis une image
Ajami déjà recadrée. C'est un checkpoint PyTorch personnalisé, pas un modèle
Transformers chargeable avec `AutoModel`. Cette publication héberge les fichiers ;
elle ne crée pas de serveur d'inférence.

Modèle et documentation :
[Hugging Face](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v9-20261015).
Archive versionnée :
[release GitHub](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/tag/v9-hausa-all-grouped-20261016).
Le nom HF conserve la date prévue du dataset ; ce modèle a été entraîné et publié
le 16 octobre 2026 avec le nouveau protocole à 5 573 images valides.
