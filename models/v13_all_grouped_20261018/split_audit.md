## Audit du split et interpretation de l'ecart validation/test

Audit du checkpoint `3b75305f19eb604974b93f47e4aeb99f020299e2ab97b7c24497b4dc004c6aac` et du split `fdb453e5f92147f31bc236998e067823b07bdce4b965811081b1d54edcaae7b5`.

Les exemples et leurs references couvrent exactement le corpus corrige, en tenant compte des exclusions documentees. Aucun identifiant ni hash de pixels exact n'est partage entre partitions. La recherche de quasi-doublons n'a pas ete realisee. Les CER, WER et UWER ont ete recalcules depuis les predictions et leurs references : ils correspondent aux fichiers de scores.

| Composition | Train | Validation | Test |
|---|---:|---:|---:|
| Segments | 4972 | 200 | 401 |
| Mots | 10657 | 447 | 2052 |
| Mots moyens par segment | 2.14 | 2.23 | 5.12 |
| Segments d'un seul mot | 3777 | 156 | 126 |
| Mots stricts absents du train (occurrences) | 0 | 144 | 505 |
| References completes deja presentes dans le train | 4972 | 89 | 65 |
| Segments de sources absentes du train | 0 | 200 | 401 |

La validation contient 78.00% de segments d'un seul mot, contre 31.42% au test. 401 exemples de test proviennent de sources absentes du train ; la repartition complete par source est dans split_audit.json. Ces differences de composition doivent etre prises en compte pour interpreter les scores de validation et du test du protocole.

| Groupe | Validation N / mots | Validation WER | Test N / mots | Test WER |
|---|---:|---:|---:|---:|
| Source presente au train | 0 / 0 | non mesure | 0 / 0 | non mesure |
| Source absente du train | 200 / 447 | 93.29% | 401 / 2052 | 86.31% |
| Segments d'un mot | 156 / 156 | 81.41% | 126 / 126 | 109.52% |
| Segments de 2 a 4 mots | 2 / 5 | 120.00% | 12 / 43 | 132.56% |
| Segments de 5 mots et plus | 42 / 286 | 99.30% | 263 / 1883 | 83.70% |
| Reference complete vue au train | 89 / 89 | 68.54% | 65 / 65 | 96.92% |
| Reference complete non vue au train | 111 / 358 | 99.44% | 336 / 1987 | 85.96% |

Ces groupes decrivent des associations, sans isoler une cause unique. La longueur ne majore pas mecaniquement le WER, qui est normalise par le nombre de mots ; elle change notamment les conditions visuelles de reconnaissance. La validation intervient aussi dans la selection du checkpoint et ne constitue pas un second test independant. Le WER des sources connues et inconnues du test doit etre examine avant d'attribuer l'ecart aux seules sources. Les conventions d'annotation et la qualite visuelle n'ont pas ete relues integralement dans cet audit.

Le split et le checkpoint documentes restent figes pour comparer V9 a V12. Les decisions de repartition reposent sur les groupes de provenance et leurs effectifs. Le fichier de protocole indique si le test public initial est conserve ou reconstruit.
