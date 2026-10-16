# V10 — corpus Hausa complet, contribution IX-World

Modèle entraîné localement le 16 octobre 2026, depuis une initialisation aléatoire, sur NVIDIA GeForce GTX 1060 6 GB. **54 époques exécutées ; checkpoint publié : époque 39**, sélectionné uniquement sur le CER de validation.

| Partition | Segments | CER | WER | UWER | Exact |
|---|---:|---:|---:|---:|---:|
| train | 4972 | 0.0021% | 0.0094% | 0.0094% | 99.9799% |
| validation | 200 | 43.5572% | 88.1432% | 86.8009% | 20.0000% |
| test | 401 | 34.8801% | 79.7758% | 79.2398% | 1.2469% |

Le test contient 401 segments et seuls 5 sont exactement reconnus. L’UWER normalise la casse et les diacritiques ; le WER strict est conservé. Les résultats des trois ensembles proviennent du même checkpoint.

Le CER de test passe de 40.6935% pour V9 à 34.8801% pour V10, soit 5.8134 points de moins sur le même test. Une seule graine a été mesurée. Le train presque parfaitement reconnu, face au WER élevé en validation et au test, indique des signes forts de surapprentissage et une généralisation insuffisante. La publication documente un résultat expérimental, sans revendication de transcription autonome fiable.

## Données, empreintes et contenu

- Dataset `IntelligenceResearchLab/Hausa`, révision `1fb85fa070b6cf6432f1d45d21f78851d57f8944`.
- Split : 4 972 train / 200 validation / 401 test ; les 5 573 images valides sont utilisées. Deux annotations contradictoires restent en quarantaine.
- Aucune image exacte, source de manuscrit ou groupe de copiste identifié partagé. Sources 1 et 2 regroupées ; les autres identités et les quasi-doublons restent à vérifier.
- Test reconstruit par sources ; il diffère du test public initial de 400 exemples.
- Code d’entraînement : `c24c44b1c4d13e0e678782e63ce0bcefd445703f`.
- Checkpoint SHA-256 : `617cd6c6a6b12503d4fd9d5541a6ec02943f1310fd70c43b447b3aaf44863357`.
- Split SHA-256 : `fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5`.
- Durée enregistrée : 466.54 minutes ; pic alloué : 5079.9 MiB.

L’archive `hausa-v10-all-grouped-20261016.zip` contient les poids, les sources exactes d’entraînement et d’évaluation, le vocabulaire, les partitions, la configuration, les prédictions, les scores, l’audit et les 54 époques. Le manifeste donne les empreintes de tous les fichiers. Aucune image du corpus n’est incluse. Le CSV, le JSON, les tableaux et les courbes PNG/SVG sont aussi proposés en téléchargements séparés.

Le checkpoint est un modèle PyTorch personnalisé ; cette publication ne crée pas de serveur d’inférence. Modèle : https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v10-20261015

## Différence actuelle entre V10 et V11

| Élément | V10 | V11 |
|---|---|---|
| Entrée | 1 canal gris | 2 canaux : gris + encre locale |
| CNN | 6 convolutions | 6 convolutions |
| BiLSTM | 2 couches, 256 unités par direction | identique |
| Supervision et décodage | CTC, décodage glouton | identique |
| Augmentation | 8 vues fixes, 39 776 vues/époque | identique |
| Batch, split, optimiseur | 24, 4 972/200/401, AdamW | identique |
| Paramètres pour 80 classes | 6 108 112 | 6 108 688 |

Le canal d’encre calcule un fond flouté gaussien (rayon 5 pixels), soustrait le gris, conserve la partie positive, applique un gain de 1,6 et normalise dans [-1,1]. La première convolution accepte ce canal supplémentaire ; les deux modèles sont entraînés indépendamment. Ce canal ne garantit pas une baisse du WER ni une correction du surapprentissage. V13 reste au stade des propositions.

## Historique des 54 époques

**54 époques exécutées ; checkpoint publié : époque 39, minimum du CER de validation.**

[CSV brut, précision d'origine](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/download/v10-hausa-all-grouped-20261016/history.csv) · [JSON avec unités et provenance](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/download/v10-hausa-all-grouped-20261016/epochs.json)

La loss du train est la moyenne des pertes CTC des batches pendant l'apprentissage avec augmentation. Les taux CER/WER/UWER et exact ci-dessous concernent uniquement la validation, sur images originales. Les CER/WER/UWER du train et du test ne sont pas enregistrés à chaque époque : ils figurent dans [metrics.csv](https://raw.githubusercontent.com/IX-World/OCR-AJAMI-Hausa/main/models/v10_all_grouped_20261016/metrics.csv), pour le checkpoint sélectionné. Aucun score manquant n'a été reconstruit.

![Courbes des mesures enregistrées par époque](https://raw.githubusercontent.com/IX-World/OCR-AJAMI-Hausa/main/models/v10_all_grouped_20261016/training_curves.png)

[Télécharger les courbes vectorielles SVG](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/download/v10-hausa-all-grouped-20261016/training_curves.svg). Courbes brutes sans lissage ; le trait pointillé indique le checkpoint sélectionné.

## Optima observés sur la validation

| Mesure | Meilleure valeur | Première époque correspondante |
|---|---:|---:|
| CER | 43.5572% | 39 |
| WER | 85.9060% | 32 |
| UWER | 84.3400% | 42 |
| Exact | 21.5000% | 33 |

Seul le minimum du CER détermine le checkpoint publié ; les autres optima ne désignent pas d'autres poids conservés. Ces scores de validation interviennent dans la sélection et ne constituent pas des résultats de test indépendants.

## Apprentissage et validation — toutes les époques

Les taux sont exprimés en pourcentage ; les valeurs exactes non arrondies sont dans le CSV et le JSON. LR utilisé = apprentissage de cette époque ; LR suivant = valeur après le scheduler.

| Époque | Loss train | VAL CER % | VAL WER % | VAL UWER % | VAL exact % | LR utilisé | LR suivant | Temps (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.451729 | 65.9710 | 98.4340 | 98.2103 | 5.0000 | 0.001 | 0.001 | 1108.063 |
| 2 | 0.812702 | 57.0327 | 93.2886 | 91.9463 | 11.5000 | 0.001 | 0.001 | 441.391 |
| 3 | 0.368961 | 50.0907 | 90.3803 | 88.5906 | 15.5000 | 0.001 | 0.001 | 511.360 |
| 4 | 0.197095 | 48.0490 | 91.2752 | 89.0380 | 13.5000 | 0.001 | 0.001 | 607.343 |
| 5 | 0.133869 | 47.7314 | 88.8143 | 86.5772 | 19.0000 | 0.001 | 0.001 | 453.094 |
| 6 | 0.082546 | 47.2323 | 87.0246 | 85.6823 | 17.0000 | 0.001 | 0.001 | 598.734 |
| 7 | 0.088119 | 46.2795 | 89.2617 | 87.6957 | 19.0000 | 0.001 | 0.001 | 866.906 |
| 8 | 0.052274 | 49.0926 | 88.1432 | 87.0246 | 20.0000 | 0.001 | 0.001 | 574.797 |
| 9 | 0.054128 | 47.4592 | 86.5772 | 84.7875 | 17.5000 | 0.001 | 0.001 | 398.484 |
| 10 | 0.034772 | 46.7332 | 90.6040 | 88.5906 | 15.0000 | 0.001 | 0.001 | 595.218 |
| 11 | 0.047147 | 47.5953 | 91.2752 | 88.3669 | 16.0000 | 0.001 | 0.0005 | 654.265 |
| 12 | 0.010313 | 45.6443 | 90.1566 | 88.3669 | 17.5000 | 0.0005 | 0.0005 | 341.656 |
| 13 | 0.008014 | 46.5064 | 89.4855 | 88.5906 | 18.0000 | 0.0005 | 0.0005 | 425.578 |
| 14 | 0.010246 | 48.1397 | 92.1700 | 90.3803 | 16.0000 | 0.0005 | 0.0005 | 340.078 |
| 15 | 0.005574 | 46.4610 | 91.0515 | 89.4855 | 18.0000 | 0.0005 | 0.0005 | 336.015 |
| 16 | 0.010087 | 47.4138 | 89.9329 | 87.6957 | 16.5000 | 0.0005 | 0.00025 | 338.484 |
| 17 | 0.002748 | 45.9165 | 90.1566 | 88.1432 | 16.5000 | 0.00025 | 0.00025 | 811.640 |
| 18 | 0.001708 | 46.4156 | 89.4855 | 87.2483 | 16.5000 | 0.00025 | 0.00025 | 350.359 |
| 19 | 0.001620 | 44.7368 | 88.1432 | 86.1298 | 17.5000 | 0.00025 | 0.00025 | 338.265 |
| 20 | 0.003639 | 46.1434 | 89.0380 | 88.1432 | 17.0000 | 0.00025 | 0.00025 | 356.219 |
| 21 | 0.001602 | 45.8711 | 89.2617 | 88.1432 | 18.0000 | 0.00025 | 0.00025 | 348.547 |
| 22 | 0.001326 | 45.5535 | 88.5906 | 87.2483 | 18.0000 | 0.00025 | 0.00025 | 461.328 |
| 23 | 0.001303 | 46.2341 | 88.5906 | 87.4720 | 18.0000 | 0.00025 | 0.000125 | 981.344 |
| 24 | 0.000646 | 45.5989 | 89.2617 | 87.9195 | 16.5000 | 0.000125 | 0.000125 | 756.735 |
| 25 | 0.000587 | 45.5535 | 87.9195 | 86.3535 | 16.5000 | 0.000125 | 0.000125 | 520.969 |
| 26 | 0.000661 | 45.6897 | 89.0380 | 87.9195 | 18.0000 | 0.000125 | 0.000125 | 667.453 |
| 27 | 0.000474 | 43.6933 | 87.4720 | 86.5772 | 20.0000 | 0.000125 | 0.000125 | 790.937 |
| 28 | 0.000539 | 45.3267 | 89.9329 | 88.8143 | 17.5000 | 0.000125 | 0.000125 | 681.187 |
| 29 | 0.000342 | 44.3285 | 87.9195 | 86.3535 | 18.5000 | 0.000125 | 0.000125 | 628.719 |
| 30 | 0.000320 | 44.6461 | 88.3669 | 86.8009 | 18.5000 | 0.000125 | 0.000125 | 872.125 |
| 31 | 0.000342 | 44.1470 | 86.3535 | 85.4586 | 20.5000 | 0.000125 | 6.25e-05 | 1350.672 |
| 32 | 0.000246 | 44.6461 | 85.9060 | 84.7875 | 18.5000 | 6.25e-05 | 6.25e-05 | 1255.312 |
| 33 | 0.000198 | 45.1452 | 86.1298 | 85.2349 | 21.5000 | 6.25e-05 | 6.25e-05 | 369.704 |
| 34 | 0.000151 | 44.2831 | 86.1298 | 85.2349 | 21.5000 | 6.25e-05 | 6.25e-05 | 335.422 |
| 35 | 0.000136 | 44.6007 | 87.2483 | 86.1298 | 19.0000 | 6.25e-05 | 3.125e-05 | 429.203 |
| 36 | 0.000105 | 44.8276 | 87.4720 | 86.3535 | 20.0000 | 3.125e-05 | 3.125e-05 | 373.172 |
| 37 | 0.000098 | 44.4192 | 86.8009 | 85.4586 | 20.5000 | 3.125e-05 | 3.125e-05 | 353.938 |
| 38 | 0.000094 | 44.5100 | 86.8009 | 85.4586 | 20.5000 | 3.125e-05 | 3.125e-05 | 336.281 |
| **39 retenue** | 0.000104 | 43.5572 | 88.1432 | 86.8009 | 20.0000 | 3.125e-05 | 3.125e-05 | 343.078 |
| 40 | 0.000094 | 44.2377 | 86.5772 | 85.4586 | 20.0000 | 3.125e-05 | 3.125e-05 | 351.828 |
| 41 | 0.000069 | 45.1452 | 87.4720 | 86.1298 | 20.0000 | 3.125e-05 | 3.125e-05 | 350.766 |
| 42 | 0.000062 | 43.7840 | 85.9060 | 84.3400 | 20.0000 | 3.125e-05 | 3.125e-05 | 339.172 |
| 43 | 0.000068 | 44.2377 | 87.0246 | 85.6823 | 20.0000 | 3.125e-05 | 1.5625e-05 | 657.547 |
| 44 | 0.000060 | 44.5554 | 85.9060 | 84.5638 | 21.5000 | 1.5625e-05 | 1.5625e-05 | 328.937 |
| 45 | 0.000050 | 44.4192 | 86.5772 | 85.2349 | 20.5000 | 1.5625e-05 | 1.5625e-05 | 340.906 |
| 46 | 0.000056 | 44.6007 | 87.0246 | 85.6823 | 20.5000 | 1.5625e-05 | 1.5625e-05 | 419.547 |
| 47 | 0.000050 | 44.8276 | 87.0246 | 85.9060 | 20.5000 | 1.5625e-05 | 7.8125e-06 | 564.422 |
| 48 | 0.000047 | 44.2377 | 86.3535 | 85.0112 | 20.5000 | 7.8125e-06 | 7.8125e-06 | 318.547 |
| 49 | 0.000047 | 44.5554 | 86.8009 | 85.4586 | 21.0000 | 7.8125e-06 | 7.8125e-06 | 329.625 |
| 50 | 0.000048 | 44.6461 | 87.4720 | 86.1298 | 19.0000 | 7.8125e-06 | 7.8125e-06 | 342.735 |
| 51 | 0.000043 | 44.1924 | 87.0246 | 85.9060 | 19.5000 | 7.8125e-06 | 3.90625e-06 | 320.266 |
| 52 | 0.000042 | 44.8730 | 87.4720 | 85.9060 | 20.0000 | 3.90625e-06 | 3.90625e-06 | 339.657 |
| 53 | 0.000041 | 44.1470 | 87.0246 | 85.6823 | 19.0000 | 3.90625e-06 | 3.90625e-06 | 342.329 |
| 54 | 0.000040 | 44.0563 | 87.4720 | 86.3535 | 18.5000 | 3.90625e-06 | 3.90625e-06 | 342.047 |

## Mémoire et évolution du CER

Mémoire en MiB : allocation mesurée en fin de validation, pic de l'époque et pic cumulé. Il s'agit de mémoire allouée par PyTorch, pas de toute la mémoire réservée ou occupée sur la carte. La pente est une régression sur au plus cinq époques récentes ; le gain est le CER du début moins celui de la fin de cette fenêtre. Une pente négative ou un gain positif indique une amélioration sur cette fenêtre.

| Époque | VRAM fin (MiB) | Pic époque (MiB) | Pic cumulé (MiB) | Pente CER (points/époque) | Gain CER (points) |
|---|---:|---:|---:|---:|---:|
| 1 | 115.584 | 5079.893 | 5079.893 | n/a | n/a |
| 2 | 114.624 | 4964.727 | 5079.893 | -8.938294 | +8.938294 |
| 3 | 114.901 | 4947.502 | 5079.893 | -7.940109 | +15.880218 |
| 4 | 115.547 | 3074.684 | 5079.893 | -6.070780 | +17.921960 |
| 5 | 115.437 | 4940.779 | 5079.893 | -4.546279 | +18.239564 |
| 6 | 116.564 | 3075.063 | 5079.893 | -2.196007 | +9.800363 |
| 7 | 115.983 | 3075.063 | 5079.893 | -0.843920 | +3.811252 |
| 8 | 117.860 | 4896.128 | 5079.893 | +0.063521 | -1.043557 |
| 9 | 119.223 | 4933.168 | 5079.893 | +0.131579 | +0.272232 |
| 10 | 115.427 | 3074.063 | 5079.893 | +0.018149 | +0.499093 |
| 11 | 114.803 | 3073.700 | 5079.893 | +0.027223 | -1.315789 |
| 12 | 118.512 | 3073.704 | 5079.893 | -0.676044 | +3.448276 |
| 13 | 114.547 | 3078.923 | 5079.893 | -0.299456 | +0.952813 |
| 14 | 115.591 | 3074.032 | 5079.893 | +0.172414 | -1.406534 |
| 15 | 117.120 | 3074.032 | 5079.893 | +0.022686 | +1.134301 |
| 16 | 115.333 | 3074.032 | 5079.893 | +0.349365 | -1.769510 |
| 17 | 115.642 | 4941.620 | 5079.893 | -0.190563 | +0.589837 |
| 18 | 115.217 | 4951.314 | 5079.893 | -0.399274 | +1.724138 |
| 19 | 118.250 | 3072.234 | 5079.893 | -0.444646 | +1.724138 |
| 20 | 121.077 | 3073.376 | 5079.893 | -0.372051 | +1.270417 |
| 21 | 114.980 | 3076.075 | 5079.893 | -0.036298 | +0.045372 |
| 22 | 118.358 | 3073.376 | 5079.893 | -0.058984 | +0.862069 |
| 23 | 115.402 | 4873.190 | 5079.893 | +0.240472 | -1.497278 |
| 24 | 115.869 | 3074.063 | 5079.893 | -0.072595 | +0.544465 |
| 25 | 117.364 | 3072.950 | 5079.893 | -0.058984 | +0.317604 |
| 26 | 116.271 | 4955.435 | 5079.893 | -0.040835 | -0.136116 |
| 27 | 114.786 | 4898.236 | 5079.893 | -0.499093 | +2.540835 |
| 28 | 115.373 | 3072.876 | 5079.893 | -0.240472 | +0.272232 |
| 29 | 119.175 | 3072.464 | 5079.893 | -0.281307 | +1.225045 |
| 30 | 115.417 | 3072.892 | 5079.893 | -0.145191 | +1.043557 |
| 31 | 118.007 | 3073.735 | 5079.893 | +0.022686 | -0.453721 |
| 32 | 115.670 | 3072.282 | 5079.893 | -0.154265 | +0.680581 |
| 33 | 114.748 | 3073.704 | 5079.893 | +0.163339 | -0.816697 |
| 34 | 114.996 | 4962.910 | 5079.893 | +0.027223 | +0.362976 |
| 35 | 115.773 | 3073.173 | 5079.893 | +0.054446 | -0.453721 |
| 36 | 114.710 | 3073.854 | 5079.893 | -0.018149 | -0.181488 |
| 37 | 117.643 | 3072.386 | 5079.893 | -0.090744 | +0.725953 |
| 38 | 114.793 | 3072.386 | 5079.893 | +0.027223 | -0.226860 |
| 39 | 117.437 | 4946.233 | 5079.893 | -0.240472 | +1.043557 |
| 40 | 114.926 | 3074.017 | 5079.893 | -0.204174 | +0.589837 |
| 41 | 115.364 | 4959.139 | 5079.893 | +0.117967 | -0.725953 |
| 42 | 115.690 | 3072.813 | 5079.893 | +0.013612 | +0.725953 |
| 43 | 116.547 | 3072.465 | 5079.893 | +0.090744 | -0.680581 |
| 44 | 114.812 | 3073.479 | 5079.893 | -0.027223 | -0.317604 |
| 45 | 116.290 | 4920.493 | 5079.893 | -0.068058 | +0.725953 |
| 46 | 114.667 | 3075.450 | 5079.893 | +0.181488 | -0.816697 |
| 47 | 114.648 | 4965.665 | 5079.893 | +0.122505 | -0.589837 |
| 48 | 122.771 | 3073.829 | 5079.893 | -0.022686 | +0.317604 |
| 49 | 114.067 | 4859.703 | 5079.893 | -0.009074 | -0.136116 |
| 50 | 120.940 | 3074.573 | 5079.893 | -0.018149 | -0.045372 |
| 51 | 114.987 | 4904.000 | 5079.893 | -0.086207 | +0.635209 |
| 52 | 115.549 | 3072.720 | 5079.893 | +0.090744 | -0.635209 |
| 53 | 117.350 | 4917.560 | 5079.893 | -0.058984 | +0.408348 |
| 54 | 117.774 | 3073.376 | 5079.893 | -0.122505 | +0.589837 |

SHA-256 du CSV d'origine : `d6f1672d7a6d7db42f59727205b51ec27cfd1b8c67777295592cfcba40267c23`. Le JSON conserve les valeurs numériques du journal, avec `null` pour les mesures non disponibles. Les temps incluent train et validation, avant la sauvegarde du checkpoint ; ils n'incluent pas le bilan final sur les trois ensembles. Une seule graine est mesurée ; les comparaisons entre versions exigent le même protocole.
