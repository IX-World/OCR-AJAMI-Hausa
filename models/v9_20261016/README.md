# IX-World — V9 initiale — résultat archivé

Ce checkpoint utilise le protocole initial 4656/517/400, séparé par images.
Des manuscrits sont partagés. Il ne satisfait pas la nouvelle campagne séparée
par provenance et ne sera pas promu comme modèle de ce nouveau protocole.
Les anciens poids restent sur la machine locale ; aucun poids n’est inclus ici.

| Partition | CER | WER | UWER |
|---|---:|---:|---:|
| Train | 0,28 % | 1,17 % | 1,14 % |
| Validation | 21,60 % | 57,58 % | 54,55 % |
| Test public | 27,07 % | 77,84 % | 76,81 % |

Checkpoint sélectionné à l’époque 71 ; fin à l’époque 79. Réévaluation CPU
strictement identique aux 400 prédictions originales sur GPU. Les scores stricts
et normalisés sont dans metrics.json ; les résultats GPU initiaux sont conservés.

Voir [l’audit](split_audit.md) et le [protocole actif](../../README_HAUSA_GROUPED_20261016.md).
La publication HF/GitHub des nouveaux modèles exige un réentraînement depuis zéro
sur le nouveau protocole, puis des résultats réels. Le dépôt HF prévu pour V9 est
IntelligenceResearchLab/Hausa-OCR-AJAMI-v9-20261015.
