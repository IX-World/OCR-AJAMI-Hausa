# Comparaison V9–V13 sur le même split — IX-World

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
