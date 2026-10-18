# V13 terminé — supervision auxiliaire AfriBERTa — IX-World

18 octobre 2026. **49 époques, checkpoint à l'époque 34.** Initialisation OCR aléatoire, enseignant figé. Test : **CER 39,0987 % / WER 86,3060 % / UWER 86,0136 %**, exact 2/401. Le surapprentissage persiste ; aucun gain sur V10/V12. Époque et localité inconnues, branches masquées : aucune capacité diachronique/régionale démontrée.

[Modèle et fiche complète sur Hugging Face](https://huggingface.co/IntelligenceResearchLab/Hausa-OCR-AJAMI-v13-20261015) · [README et intégrité dans Git](https://github.com/IX-World/OCR-AJAMI-Hausa/tree/main/models/v13_all_grouped_20261018)

## Comparaison V9–V13 sur le même split — IX-World

Protocole commun : 4 972 train / 200 validation / 401 test, mêmes identifiants, vocabulaire et graine 42. Checkpoint de chaque modèle sélectionné sur son CER de validation.

| Modèle | Époque retenue | Époques exécutées | Test CER | Test WER | Test UWER | Segments exacts |
|---|---:|---:|---:|---:|---:|---:|
| V9 | 33 | 48 | 40.6935% | 86.9396% | 86.1598% | 5/401 |
| V10 | 39 | 54 | 34.8801% | 79.7758% | 79.2398% | 5/401 |
| V11 | 39 | 54 | 35.7136% | 83.4795% | 82.6998% | 9/401 |
| V12 | 50 | 65 | 33.5425% | 79.6784% | 78.8499% | 5/401 |
| V13 | 34 | 49 | 39.0987% | 86.3060% | 86.0136% | 2/401 |

**V13 n'apporte pas le gain attendu sur ce run.**

Par rapport à V10, V13 augmente le CER de **4.2185 points** (410 erreurs de caractères supplémentaires sur 9 719) et le WER de **6.5302 points** (134 erreurs de mots supplémentaires sur 2 052).

Par rapport à V12, V13 augmente le CER de **5.5561 points** (540 erreurs de caractères supplémentaires sur 9 719) et le WER de **6.6277 points** (136 erreurs de mots supplémentaires sur 2 052).

V13 reconnaît presque parfaitement le train, mais obtient 44,37 % de CER et 93,29 % de WER de validation ; le surapprentissage persiste. Les objectifs CER ≤ 20 % et WER ≤ 30 % ne sont pas atteints. V12 reste le meilleur CER de test observé dans cette série, sans preuve de supériorité générale ni de significativité : une seule graine est mesurée.

V13 est parti de poids OCR aléatoires, sans initialisation V10. L'enseignant AfriBERTa est préentraîné et figé ; ses cibles sont calculées exclusivement depuis les références du train. Langue haoussa renseignée, époque/localité inconnues et masquées, zéro contexte textuel complémentaire vérifié : aucun résultat de compréhension historique ou régionale n'est établi.

V10 : six convolutions gris, BatchNorm, BiLSTM non empaqueté. V11 ajoute le canal d'encre. V12 utilise neuf convolutions et deux canaux, avec dropout CNN 0,05 et batch 20. V13 : six convolutions gris, GroupNorm, séquences empaquetées, dropout 0,1/0,2, batch jusqu'à 6 et accumulation 4, LR 0,0005, weight decay 0,001, perte auxiliaire sémantique. Plusieurs facteurs changent : ni le bénéfice ni la régression ne peuvent être attribués à la seule sémantique sans ablation λ=0 et répétitions.

Les scores de test documentent les expériences terminées ; les choix suivants doivent reposer sur la validation. Les consultations répétées du test dans cette campagne limitent son rôle pour une affirmation finale indépendante. Les courbes par époque contiennent les scores de validation et les pertes train, pas des scores de test par époque. Les dénominateurs et compteurs sont dans comparison_v9_v10_v11_v12_v13.json.

# Valeurs enregistrées à chaque époque

**49 époques exécutées ; checkpoint publié : époque 34, minimum du CER de validation.**

[CSV brut, précision d'origine](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/download/v13-hausa-all-grouped-20261018/history.csv) · [JSON avec unités et provenance](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/download/v13-hausa-all-grouped-20261018/epochs.json)

La loss train de V13 combine CTC + λ × perte sémantique. Les pertes sont moyennées en pondérant chaque microbatch par son nombre d'exemples augmentés. La perte sémantique vaut 0,5 × cosinus + 0,5 × contrastive ; ses négatifs proviennent du microbatch réel, pas du batch accumulé. Les taux CER/WER/UWER et exact ci-dessous concernent uniquement la validation, sur images originales. Les CER/WER/UWER du train et du test ne sont pas enregistrés à chaque époque : ils figurent dans [metrics.csv](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/download/v13-hausa-all-grouped-20261018/metrics.csv), pour le checkpoint sélectionné. Aucun score manquant n'a été reconstruit.

![Courbes des mesures enregistrées par époque](https://raw.githubusercontent.com/IX-World/OCR-AJAMI-Hausa/main/models/v13_all_grouped_20261018/training_curves.png)

[Télécharger les courbes vectorielles SVG](https://github.com/IX-World/OCR-AJAMI-Hausa/releases/download/v13-hausa-all-grouped-20261018/training_curves.svg). Courbes brutes sans lissage ; le trait pointillé indique le checkpoint sélectionné.

## Optima observés sur la validation

| Mesure | Meilleure valeur | Première époque correspondante |
|---|---:|---:|
| CER | 44.3739% | 34 |
| WER | 89.9329% | 7 |
| UWER | 88.8143% | 7 |
| Exact | 17.5000% | 7 |

Seul le minimum du CER détermine le checkpoint publié ; les autres optima ne désignent pas d'autres poids conservés. Ces scores de validation interviennent dans la sélection et ne constituent pas des résultats de test indépendants.

## Apprentissage et validation — toutes les époques

Les taux sont exprimés en pourcentage ; les valeurs exactes non arrondies sont dans le CSV et le JSON. LR utilisé = apprentissage de cette époque ; LR suivant = valeur après le scheduler.

| Époque | Loss train | VAL CER % | VAL WER % | VAL UWER % | VAL exact % | LR utilisé | LR suivant | Temps (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.966864 | 87.3412 | 97.5391 | 97.0917 | 5.5000 | 0.0005 | 0.0005 | 606.594 |
| 2 | 2.372565 | 68.9201 | 98.8814 | 98.6577 | 3.0000 | 0.0005 | 0.0005 | 585.484 |
| 3 | 1.378943 | 56.5789 | 94.8546 | 93.9597 | 12.0000 | 0.0005 | 0.0005 | 581.875 |
| 4 | 0.780955 | 51.9510 | 93.5123 | 92.8412 | 11.5000 | 0.0005 | 0.0005 | 595.281 |
| 5 | 0.490081 | 50.3630 | 92.1700 | 91.2752 | 14.0000 | 0.0005 | 0.0005 | 590.875 |
| 6 | 0.330056 | 50.1361 | 94.6309 | 92.8412 | 9.5000 | 0.0005 | 0.0005 | 595.797 |
| 7 | 0.250772 | 47.1869 | 89.9329 | 88.8143 | 17.5000 | 0.0005 | 0.0005 | 623.109 |
| 8 | 0.195249 | 47.6407 | 92.6174 | 91.4989 | 13.0000 | 0.0005 | 0.0005 | 596.079 |
| 9 | 0.156030 | 48.3666 | 94.4072 | 93.0649 | 10.0000 | 0.0005 | 0.0005 | 587.609 |
| 10 | 0.126898 | 47.5499 | 92.6174 | 91.0515 | 12.0000 | 0.0005 | 0.0005 | 586.719 |
| 11 | 0.120564 | 48.6842 | 92.6174 | 91.4989 | 14.0000 | 0.0005 | 0.00025 | 581.313 |
| 12 | 0.066071 | 46.8240 | 92.8412 | 90.3803 | 15.0000 | 0.00025 | 0.00025 | 588.844 |
| 13 | 0.051797 | 46.1887 | 91.9463 | 90.6040 | 16.0000 | 0.00025 | 0.00025 | 596.625 |
| 14 | 0.052857 | 45.6443 | 91.2752 | 89.2617 | 15.5000 | 0.00025 | 0.00025 | 600.391 |
| 15 | 0.045267 | 45.5082 | 93.2886 | 91.9463 | 16.0000 | 0.00025 | 0.00025 | 607.047 |
| 16 | 0.047208 | 45.2359 | 90.6040 | 89.0380 | 16.0000 | 0.00025 | 0.00025 | 567.812 |
| 17 | 0.041286 | 45.7350 | 91.9463 | 89.9329 | 15.5000 | 0.00025 | 0.00025 | 563.485 |
| 18 | 0.035010 | 49.1379 | 98.8814 | 96.6443 | 12.0000 | 0.00025 | 0.00025 | 566.078 |
| 19 | 0.045844 | 46.6425 | 91.4989 | 89.0380 | 14.5000 | 0.00025 | 0.00025 | 585.172 |
| 20 | 0.032357 | 46.4156 | 94.4072 | 91.9463 | 12.5000 | 0.00025 | 0.000125 | 606.563 |
| 21 | 0.028875 | 44.9183 | 92.6174 | 90.8277 | 14.0000 | 0.000125 | 0.000125 | 563.703 |
| 22 | 0.026632 | 45.2813 | 91.4989 | 90.1566 | 15.0000 | 0.000125 | 0.000125 | 567.484 |
| 23 | 0.026996 | 44.7822 | 93.9597 | 91.9463 | 15.5000 | 0.000125 | 0.000125 | 570.062 |
| 24 | 0.025315 | 46.4610 | 93.5123 | 91.0515 | 15.0000 | 0.000125 | 0.000125 | 563.984 |
| 25 | 0.024813 | 45.5989 | 93.5123 | 91.4989 | 15.5000 | 0.000125 | 0.000125 | 565.578 |
| 26 | 0.023709 | 45.3267 | 91.0515 | 89.2617 | 15.0000 | 0.000125 | 0.000125 | 565.797 |
| 27 | 0.023401 | 45.1452 | 92.1700 | 90.1566 | 14.5000 | 0.000125 | 6.25e-05 | 564.296 |
| 28 | 0.021673 | 44.6007 | 92.6174 | 91.0515 | 15.5000 | 6.25e-05 | 6.25e-05 | 566.781 |
| 29 | 0.021051 | 45.2813 | 93.5123 | 91.7226 | 16.0000 | 6.25e-05 | 6.25e-05 | 563.281 |
| 30 | 0.020608 | 45.4628 | 93.0649 | 91.0515 | 14.5000 | 6.25e-05 | 6.25e-05 | 571.234 |
| 31 | 0.020575 | 45.1452 | 93.2886 | 91.4989 | 14.5000 | 6.25e-05 | 6.25e-05 | 562.578 |
| 32 | 0.020143 | 45.1906 | 92.8412 | 90.8277 | 16.0000 | 6.25e-05 | 3.125e-05 | 584.516 |
| 33 | 0.019365 | 45.0544 | 94.1834 | 92.3937 | 15.0000 | 3.125e-05 | 3.125e-05 | 565.828 |
| **34 retenue** | 0.019170 | 44.3739 | 93.2886 | 91.4989 | 14.5000 | 3.125e-05 | 3.125e-05 | 563.844 |
| 35 | 0.019036 | 44.5554 | 92.8412 | 90.6040 | 16.0000 | 3.125e-05 | 3.125e-05 | 566.250 |
| 36 | 0.018824 | 45.0998 | 93.2886 | 91.2752 | 15.0000 | 3.125e-05 | 3.125e-05 | 562.875 |
| 37 | 0.018637 | 44.7368 | 93.7360 | 91.9463 | 14.5000 | 3.125e-05 | 3.125e-05 | 566.860 |
| 38 | 0.018448 | 45.6443 | 94.4072 | 92.1700 | 13.5000 | 3.125e-05 | 1.5625e-05 | 564.563 |
| 39 | 0.018306 | 45.1906 | 93.2886 | 91.4989 | 14.5000 | 1.5625e-05 | 1.5625e-05 | 566.531 |
| 40 | 0.018170 | 45.2813 | 93.7360 | 91.9463 | 15.0000 | 1.5625e-05 | 1.5625e-05 | 567.641 |
| 41 | 0.018171 | 44.9183 | 93.7360 | 91.7226 | 15.0000 | 1.5625e-05 | 1.5625e-05 | 568.641 |
| 42 | 0.018073 | 45.1906 | 94.8546 | 92.6174 | 14.5000 | 1.5625e-05 | 7.8125e-06 | 568.375 |
| 43 | 0.017880 | 44.9183 | 93.2886 | 91.4989 | 14.5000 | 7.8125e-06 | 7.8125e-06 | 564.297 |
| 44 | 0.017819 | 44.8730 | 93.9597 | 91.7226 | 15.0000 | 7.8125e-06 | 7.8125e-06 | 577.578 |
| 45 | 0.017722 | 44.8276 | 93.5123 | 91.4989 | 16.0000 | 7.8125e-06 | 7.8125e-06 | 567.485 |
| 46 | 0.017749 | 44.9637 | 93.2886 | 91.4989 | 15.5000 | 7.8125e-06 | 3.90625e-06 | 566.328 |
| 47 | 0.017633 | 44.8276 | 93.2886 | 91.2752 | 15.0000 | 3.90625e-06 | 3.90625e-06 | 569.016 |
| 48 | 0.017633 | 44.5554 | 93.9597 | 92.1700 | 15.0000 | 3.90625e-06 | 3.90625e-06 | 581.219 |
| 49 | 0.017587 | 44.8730 | 93.7360 | 91.7226 | 15.0000 | 3.90625e-06 | 3.90625e-06 | 572.500 |

## Décomposition des pertes V13

Les pertes sémantiques sont calculées même lorsque λ = 0 ; elles ne contribuent alors pas au gradient du réseau. Mises à jour = appels réels à optimizer.step après accumulation.

| Époque | CTC | Sémantique | Cosinus | Contrastive | λ | Mises à jour |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2.966864 | 1.381816 | 1.011722 | 1.751911 | 0.0000 | 1658 |
| 2 | 2.372565 | 1.378461 | 1.004111 | 1.752811 | 0.0000 | 1658 |
| 3 | 1.378943 | 1.374859 | 0.994147 | 1.755571 | 0.0000 | 1658 |
| 4 | 0.769240 | 0.585740 | 0.295570 | 0.875911 | 0.0200 | 1658 |
| 5 | 0.471171 | 0.472754 | 0.288585 | 0.656924 | 0.0400 | 1658 |
| 6 | 0.304614 | 0.424031 | 0.280294 | 0.567768 | 0.0600 | 1658 |
| 7 | 0.219135 | 0.395464 | 0.273358 | 0.517570 | 0.0800 | 1658 |
| 8 | 0.158287 | 0.369614 | 0.265233 | 0.473996 | 0.1000 | 1658 |
| 9 | 0.120812 | 0.352190 | 0.259405 | 0.444974 | 0.1000 | 1658 |
| 10 | 0.093081 | 0.338170 | 0.253663 | 0.422678 | 0.1000 | 1658 |
| 11 | 0.087241 | 0.333227 | 0.251880 | 0.414575 | 0.1000 | 1658 |
| 12 | 0.036238 | 0.298338 | 0.235576 | 0.361099 | 0.1000 | 1658 |
| 13 | 0.023714 | 0.280824 | 0.226389 | 0.335259 | 0.1000 | 1658 |
| 14 | 0.025096 | 0.277614 | 0.224756 | 0.330473 | 0.1000 | 1658 |
| 15 | 0.018568 | 0.266986 | 0.218785 | 0.315187 | 0.1000 | 1658 |
| 16 | 0.020663 | 0.265455 | 0.218304 | 0.312607 | 0.1000 | 1658 |
| 17 | 0.015659 | 0.256264 | 0.212799 | 0.299729 | 0.1000 | 1658 |
| 18 | 0.010422 | 0.245881 | 0.206604 | 0.285159 | 0.1000 | 1658 |
| 19 | 0.020296 | 0.255479 | 0.212950 | 0.298008 | 0.1000 | 1658 |
| 20 | 0.008527 | 0.238295 | 0.202150 | 0.274441 | 0.1000 | 1658 |
| 21 | 0.006136 | 0.227384 | 0.194567 | 0.260201 | 0.1000 | 1658 |
| 22 | 0.004596 | 0.220360 | 0.189531 | 0.251189 | 0.1000 | 1658 |
| 23 | 0.005186 | 0.218107 | 0.188174 | 0.248041 | 0.1000 | 1658 |
| 24 | 0.004105 | 0.212106 | 0.184030 | 0.240182 | 0.1000 | 1658 |
| 25 | 0.003966 | 0.208466 | 0.181493 | 0.235440 | 0.1000 | 1658 |
| 26 | 0.003297 | 0.204115 | 0.178252 | 0.229978 | 0.1000 | 1658 |
| 27 | 0.003244 | 0.201575 | 0.176120 | 0.227030 | 0.1000 | 1658 |
| 28 | 0.002163 | 0.195104 | 0.171516 | 0.218691 | 0.1000 | 1658 |
| 29 | 0.001882 | 0.191692 | 0.168728 | 0.214656 | 0.1000 | 1658 |
| 30 | 0.001729 | 0.188788 | 0.166615 | 0.210960 | 0.1000 | 1658 |
| 31 | 0.001859 | 0.187161 | 0.165257 | 0.209065 | 0.1000 | 1658 |
| 32 | 0.001666 | 0.184772 | 0.163458 | 0.206087 | 0.1000 | 1658 |
| 33 | 0.001284 | 0.180809 | 0.160763 | 0.200856 | 0.1000 | 1658 |
| 34 | 0.001207 | 0.179627 | 0.159565 | 0.199688 | 0.1000 | 1658 |
| 35 | 0.001154 | 0.178814 | 0.158580 | 0.199049 | 0.1000 | 1658 |
| 36 | 0.001099 | 0.177255 | 0.157494 | 0.197016 | 0.1000 | 1658 |
| 37 | 0.001074 | 0.175634 | 0.156216 | 0.195053 | 0.1000 | 1658 |
| 38 | 0.001045 | 0.174031 | 0.155182 | 0.192880 | 0.1000 | 1658 |
| 39 | 0.000954 | 0.173517 | 0.154195 | 0.192839 | 0.1000 | 1658 |
| 40 | 0.000925 | 0.172442 | 0.153467 | 0.191417 | 0.1000 | 1658 |
| 41 | 0.000954 | 0.172171 | 0.153123 | 0.191219 | 0.1000 | 1658 |
| 42 | 0.000914 | 0.171592 | 0.152507 | 0.190676 | 0.1000 | 1658 |
| 43 | 0.000875 | 0.170052 | 0.151593 | 0.188511 | 0.1000 | 1658 |
| 44 | 0.000832 | 0.169864 | 0.151581 | 0.188148 | 0.1000 | 1658 |
| 45 | 0.000798 | 0.169243 | 0.151034 | 0.187452 | 0.1000 | 1658 |
| 46 | 0.000822 | 0.169273 | 0.150790 | 0.187756 | 0.1000 | 1658 |
| 47 | 0.000768 | 0.168644 | 0.150571 | 0.186717 | 0.1000 | 1658 |
| 48 | 0.000783 | 0.168505 | 0.150299 | 0.186711 | 0.1000 | 1658 |
| 49 | 0.000746 | 0.168409 | 0.150175 | 0.186643 | 0.1000 | 1658 |

## Mémoire et évolution du CER

Mémoire en MiB : allocation mesurée en fin de validation, pic de l'époque et pic cumulé. Il s'agit de mémoire allouée par PyTorch, pas de toute la mémoire réservée ou occupée sur la carte. La pente est une régression sur au plus cinq époques récentes ; le gain est le CER du début moins celui de la fin de cette fenêtre. Une pente négative ou un gain positif indique une amélioration sur cette fenêtre.

| Époque | VRAM fin (MiB) | Pic époque (MiB) | Pic cumulé (MiB) | Pente CER (points/époque) | Gain CER (points) |
|---|---:|---:|---:|---:|---:|
| 1 | 119.354 | 884.515 | 884.515 | n/a | n/a |
| 2 | 119.557 | 880.359 | 884.515 | -18.421053 | +18.421053 |
| 3 | 118.635 | 884.731 | 884.731 | -15.381125 | +30.762250 |
| 4 | 119.307 | 888.531 | 888.531 | -11.851180 | +35.390200 |
| 5 | 119.197 | 883.085 | 888.531 | -9.092559 | +36.978221 |
| 6 | 119.807 | 886.133 | 888.531 | -4.378403 | +18.784029 |
| 7 | 119.697 | 884.847 | 888.531 | -2.059891 | +9.392015 |
| 8 | 119.105 | 885.518 | 888.531 | -1.179673 | +4.310345 |
| 9 | 118.666 | 884.104 | 888.531 | -0.648820 | +1.996370 |
| 10 | 120.408 | 886.987 | 888.531 | -0.399274 | +2.586207 |
| 11 | 119.611 | 887.218 | 888.531 | +0.290381 | -1.497278 |
| 12 | 120.414 | 883.795 | 888.531 | -0.131579 | +0.816697 |
| 13 | 119.688 | 883.518 | 888.531 | -0.508167 | +2.177858 |
| 14 | 119.166 | 891.645 | 891.645 | -0.630672 | +1.905626 |
| 15 | 119.128 | 885.791 | 891.645 | -0.753176 | +3.176044 |
| 16 | 119.072 | 885.300 | 891.645 | -0.385662 | +1.588022 |
| 17 | 118.635 | 888.374 | 891.645 | -0.131579 | +0.453721 |
| 18 | 118.635 | 888.683 | 891.645 | +0.721416 | -3.493648 |
| 19 | 118.765 | 876.565 | 891.645 | +0.617060 | -1.134301 |
| 20 | 120.035 | 886.757 | 891.645 | +0.326679 | -1.179673 |
| 21 | 120.101 | 886.085 | 891.645 | -0.435572 | +0.816697 |
| 22 | 119.272 | 887.807 | 891.645 | -0.943739 | +3.856624 |
| 23 | 119.197 | 877.030 | 891.645 | -0.485481 | +1.860254 |
| 24 | 119.836 | 884.274 | 891.645 | -0.004537 | -0.045372 |
| 25 | 121.041 | 887.385 | 891.645 | +0.254083 | -0.680581 |
| 26 | 119.629 | 888.857 | 891.645 | +0.090744 | -0.045372 |
| 27 | 119.018 | 883.949 | 891.645 | -0.040835 | -0.362976 |
| 28 | 119.385 | 885.737 | 891.645 | -0.417423 | +1.860254 |
| 29 | 119.432 | 882.677 | 891.645 | -0.136116 | +0.317604 |
| 30 | 119.693 | 886.529 | 891.645 | +0.040835 | -0.136116 |
| 31 | 119.058 | 889.820 | 891.645 | +0.086207 | +0.000000 |
| 32 | 119.682 | 884.793 | 891.645 | +0.104356 | -0.589837 |
| 33 | 119.406 | 891.543 | 891.645 | -0.072595 | +0.226860 |
| 34 | 119.257 | 883.076 | 891.645 | -0.226860 | +1.088929 |
| 35 | 119.342 | 892.252 | 892.252 | -0.199637 | +0.589837 |
| 36 | 120.072 | 883.851 | 892.252 | -0.068058 | +0.090744 |
| 37 | 118.874 | 882.896 | 892.252 | +0.009074 | +0.317604 |
| 38 | 119.568 | 888.717 | 892.252 | +0.272232 | -1.270417 |
| 39 | 119.314 | 885.230 | 892.252 | +0.181488 | -0.635209 |
| 40 | 119.005 | 885.170 | 892.252 | +0.081670 | -0.181488 |
| 41 | 119.150 | 884.974 | 892.252 | +0.000000 | -0.181488 |
| 42 | 119.193 | 887.691 | 892.252 | -0.117967 | +0.453721 |
| 43 | 119.583 | 883.871 | 892.252 | -0.063521 | +0.272232 |
| 44 | 118.635 | 881.392 | 892.252 | -0.081670 | +0.408348 |
| 45 | 119.564 | 887.724 | 892.252 | -0.049909 | +0.090744 |
| 46 | 120.119 | 883.316 | 892.252 | -0.054446 | +0.226860 |
| 47 | 119.307 | 884.605 | 892.252 | -0.009074 | +0.090744 |
| 48 | 118.756 | 885.929 | 892.252 | -0.063521 | +0.317604 |
| 49 | 121.135 | 882.250 | 892.252 | -0.031760 | -0.045372 |

SHA-256 du CSV d'origine : `81faf1351f2559baefc51c653d75e504cfad053ef71859670cb532e3096e80a9`. Le JSON conserve les valeurs numériques du journal, avec `null` pour les mesures non disponibles. Les temps incluent train et validation, avant la sauvegarde du checkpoint ; ils n'incluent pas le bilan final sur les trois ensembles. Une seule graine est mesurée ; les comparaisons entre versions exigent le même protocole.
