## Audit du split et interpretation de l'ecart validation/test

Audit du checkpoint `02e84bffcff1293c0b17523c1dca3f0dddb8b12d55c0ed878d6a9261f084cc1d` et du split `6548df277ff5ebbb491f208eae7f14e7802e948b96d045545f8ae0009a1d39cb`.

Les exemples et leurs references couvrent exactement le corpus corrige, en tenant compte des exclusions documentees. Aucun identifiant ni hash de pixels exact n'est partage entre partitions. La recherche de quasi-doublons n'a pas ete realisee. Les CER, WER et UWER ont ete recalcules depuis les predictions et leurs references : ils correspondent aux fichiers de scores.

| Composition | Train | Validation | Test |
|---|---:|---:|---:|
| Segments | 4656 | 517 | 400 |
| Mots | 9809 | 1122 | 2225 |
| Mots moyens par segment | 2.11 | 2.17 | 5.56 |
| Segments d'un seul mot | 3581 | 397 | 81 |
| Mots stricts absents du train (occurrences) | 0 | 140 | 353 |
| References completes deja presentes dans le train | 4656 | 274 | 55 |
| Segments de sources absentes du train | 0 | 0 | 216 |

La validation contient 76.79% de segments d'un seul mot, contre 20.25% au test. 216 exemples de test proviennent de sources absentes du train ; la repartition complete par source est dans split_audit.json. La validation ne constitue donc pas un echantillon representatif de toutes les conditions du test public.

| Groupe | Validation N / mots | Validation WER | Test N / mots | Test WER |
|---|---:|---:|---:|---:|
| Source presente au train | 517 / 1122 | 57.58% | 184 / 693 | 77.06% |
| Source absente du train | 0 / 0 | non mesure | 216 / 1532 | 78.20% |
| Segments d'un mot | 397 / 397 | 43.07% | 81 / 81 | 46.91% |
| Segments de 2 a 4 mots | 17 / 60 | 88.33% | 11 / 41 | 121.95% |
| Segments de 5 mots et plus | 103 / 665 | 63.46% | 308 / 2103 | 78.17% |
| Reference complete vue au train | 274 / 274 | 28.83% | 55 / 55 | 34.55% |
| Reference complete non vue au train | 243 / 848 | 66.86% | 345 / 2170 | 78.94% |

Ces groupes decrivent des associations, sans isoler une cause unique. La longueur ne majore pas mecaniquement le WER, qui est normalise par le nombre de mots ; elle change notamment les conditions visuelles de reconnaissance. La validation intervient aussi dans la selection du checkpoint et ne constitue pas un second test independant. Le WER des sources connues et inconnues du test doit etre examine avant d'attribuer l'ecart aux seules sources. Les conventions d'annotation et la qualite visuelle n'ont pas ete relues integralement dans cet audit.

Le split et le checkpoint documentes restent figes pour comparer V9 a V12. Les decisions de repartition reposent sur les groupes de provenance et leurs effectifs. Le fichier de protocole indique si le test public initial est conserve ou reconstruit.
