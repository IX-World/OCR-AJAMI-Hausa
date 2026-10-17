# Comparaison V9–V12 sur le même split — IX-World

Protocole commun : 4 972 train / 200 validation / 401 test, mêmes identifiants et vocabulaire, graine 42. Chaque checkpoint est sélectionné sur son CER de validation.

| Modèle | Époque retenue | Époques exécutées | Test CER | Test WER | Test UWER | Segments exacts |
|---|---:|---:|---:|---:|---:|---:|
| V9 | 33 | 48 | 40.6935% | 86.9396% | 86.1598% | 1.2469% |
| V10 | 39 | 54 | 34.8801% | 79.7758% | 79.2398% | 1.2469% |
| V11 | 39 | 54 | 35.7136% | 83.4795% | 82.6998% | 2.2444% |
| V12 | 50 | 65 | 33.5425% | 79.6784% | 78.8499% | 1.2469% |

V12 réduit le CER de test de 1.3376 point par rapport à V10 : 130 erreurs de caractères en moins sur 9719 caractères. Le WER baisse de 0.0975 point, soit seulement 2 erreurs de mots en moins sur 2052 mots. Il reste élevé. Le nombre de segments exactement reconnus reste de 5 sur 401.

Le train de V12 est reconnu sans erreur, face à 41,97 % de CER de validation et 33,54 % de CER de test. L'écart de généralisation reste important. Les objectifs CER ≤ 20 % et WER ≤ 30 % ne sont pas atteints. Une seule graine ne permet pas de conclure à une supériorité générale ni à la significativité des écarts.

V10 : six convolutions, un canal gris, batch 24. V11 : six convolutions, gris + encre, batch 24. V12 : neuf convolutions, gris + encre, dropout CNN 0,05, batch 20 et organisation des poolings différente. L'effet de la seule profondeur n'est donc pas isolé. Les autres protocoles et anciens tests ne sont pas directement comparables à cette campagne.

Les scores de test sont rapportés pour documenter ces expériences terminées. Les réglages des expériences suivantes doivent être sélectionnés sur la validation. Les dénominateurs, compteurs et empreintes sont dans comparison_v9_v10_v11_v12.json. Les CER/WER/UWER de train et test ne sont pas disponibles à chaque époque : seul leur bilan final est mesuré.
