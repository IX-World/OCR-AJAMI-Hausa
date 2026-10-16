# Campagne Hausa du 15 octobre 2026 — IX-World

Ce code reprend les architectures et les augmentations V9/V10/V11/V12 du projet
[IX-World/OCR-AJAMI-Hausa](https://github.com/IX-World/OCR-AJAMI-Hausa).
Les expériences utilisent les nouvelles données, des chemins sous `D:\ai_test`,
un même split public et une recette d'optimisation commune. Les anciens
checkpoints et leurs résultats ne sont pas remplacés par des résultats non calculés.

## Données corrigées et split

Dataset : `IntelligenceResearchLab/Hausa`, révision
`1fb85fa070b6cf6432f1d45d21f78851d57f8944`.

| Partition | Segments réels | Origine |
|---|---:|---|
| Train | 4 656 | Train public corrigé |
| Validation | 517 | 10 % du train public corrigé, graine 42, stratifié par source |
| Test | 400 | Test public conservé, jamais ajouté au train ou à la validation |

Les corrections sont documentées dans `protocol_hausa_20261015/protocol.json` :

- `ajami_00462png` → `ajami_00462.png` ;
- `ajami_01243png` → `ajami_01243.png` ;
- `ajami_01O53.png` → `ajami_01053.png` ;
- l'annotation « ranar » liée à tort à `ajami_01299.png` est rattachée à
  `ajami_01289.png`, selon la séquence 01288/01289/01290 et les images vérifiées ;
- une répétition identique de `c_ajami_00147.png` est supprimée ;
- `a_ajami_03400.png` et `a_ajami_03401.png` ont les mêmes pixels mais les
  annotations contradictoires « da » et « mutane » : les deux sont exclues sur
  décision de l'utilisateur, sans inventer une transcription. Elles restent
  sur disque et dans `quarantine.json` pour vérification linguistique.

Le fichier original de métadonnées est sauvegardé avant modification.
Les métadonnées du test restent identiques octet par octet. Aucune image n'est
supprimée. Les manifestes contiennent les empreintes SHA-256 des images,
des pixels, du vocabulaire et du split ; aucun doublon exact de pixels ne
traverse les partitions. Les quasi-doublons ne sont pas contrôlés, et cette
validation n'est pas une évaluation sur manuscrits ou copistes inconnus.

Le vocabulaire est calculé sur le train uniquement, avec BLANK=0 et UNK=1.
Les apostrophes ASCII et les crochets `[` et `]` présents dans le test ne
figurent pas dans le train ; cette limite est consignée, sans modifier les
références du test pour améliorer artificiellement le score.

## Dossiers et Python local

```text
D:\ai_test\
  Hausa_repo_2026-10-15\data\train\metadata.jsonl
  Hausa_repo_2026-10-15\data\test\metadata.jsonl
  protocol_hausa_20261015\split.json
  OCR-AJAMI-Hausa\train_hausa.py
  runs_hausa_20261015\v9_seed42\
  runs_hausa_20261015\v10_seed42\
  runs_hausa_20261015\v11_seed42\
  runs_hausa_20261015\v12_seed42\
```

`python_ocr.cmd` emploie en priorité `D:\ai_test\.venv-hausa`, ou le chemin
défini dans `HAUSA_PYTHON`. Sur la machine préparée, il peut réutiliser le Python
CUDA déjà validé dans `C:\Users\admin\Documents\ChatGPT\AJAMI\.venv-v10`.
L'exécution a lieu dans votre terminal Windows ; Codex n'est pas nécessaire.
Ce chemin de secours doit être changé sur une autre machine.

Pour créer, au besoin, un environnement indépendant dans D: :

```cmd
cd /d D:\ai_test
py -3.12 -m venv .venv-hausa
.venv-hausa\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv-hausa\Scripts\python.exe -m pip install -r OCR-AJAMI-Hausa\requirements-hausa.txt
```

## Lancer les quatre modèles dans l'ordre

Dans **Invite de commandes Windows (cmd)** :

```cmd
cd /d D:\ai_test
run_v9_v12.cmd
```

Le raccourci dans D: appelle le script du dépôt ; les quatre modèles s'exécutent
successivement sur la GTX 1060, et non simultanément. Le test public est calculé
automatiquement après chaque train, en rechargeant le meilleur CER de validation.
Le lancement s'arrête si un modèle rencontre une erreur.

Dans **PowerShell** :

```powershell
Set-Location D:\ai_test
.\run_v9_v12.cmd
```

Pour un seul modèle depuis le dépôt :

```cmd
cd /d D:\ai_test\OCR-AJAMI-Hausa
python_ocr.cmd --version v9
python_ocr.cmd --version v10
python_ocr.cmd --version v11
python_ocr.cmd --version v12
```

`Ctrl+C` interrompt le modèle courant. Après au moins une époque complète :

```cmd
python_ocr.cmd --version v11 --resume
```

Une époque partielle n'est pas sauvegardée ; la reprise repart du dernier
checkpoint d'époque complète avec les états de l'optimiseur, du scheduler et
des générateurs pseudo-aléatoires. Un dossier contenant déjà un run ne peut
pas être réutilisé silencieusement : employer `--resume` ou un autre `--output`.

Pour refaire uniquement les tests, à partir des checkpoints déjà entraînés :

```cmd
cd /d D:\ai_test
test_v9_v12.cmd
```

Ou `python_ocr.cmd --version v12 --test-only` depuis le dépôt.
Un test est refusé si le checkpoint, le split, le vocabulaire ou les données
ne correspondent plus au run. Les erreurs non finies CTC ne sont pas masquées.

### CMD modèle par modèle depuis D:\ai_test

```cmd
cd /d D:\ai_test
lancer_v9.cmd
lancer_v10.cmd
lancer_v11.cmd
lancer_v12.cmd
```

Exécuter une seule ligne `lancer_...` à la fois. Chacune effectue le train puis
le test de sa version. Pour reprendre, par exemple : `lancer_v10.cmd --resume`.
Pour le test seul : `tester_v9.cmd`, `tester_v10.cmd`, `tester_v11.cmd` ou `tester_v12.cmd`.

### Notebooks dédiés

Les quatre notebooks à jour se trouvent dans `notebooks/` :
`v9_hausa_20261015.ipynb`, `v10_hausa_20261015.ipynb`,
`v11_hausa_20261015.ipynb` et `v12_hausa_20261015.ipynb`.
Ils emploient les mêmes scripts, images et identifiants de split que les CMD.
Le paramètre `ACTION` permet de choisir `check`, `train`, `resume` ou `test`.
Ils n'ont pas de sorties de résultats préremplies. Le packaging est désactivé
par défaut ; aucune publication HF n'est lancée par leur exécution normale.

## Configurations comparées

| Version | CNN | Entrée | Augmentation | Batch par défaut |
|---|---|---|---|---:|
| V9 | 6 convolutions | Gris | Dynamique, probabilité 0,30 | 24 |
| V10 | 6 convolutions | Gris | 8 vues fixes | 24 |
| V11 | 6 convolutions | Gris + encre locale | 8 vues fixes | 24 |
| V12 | 9 convolutions, dropout spatial 0,05 | Gris + encre locale | 8 vues fixes | 20 |

BiLSTM commun : deux couches, 256 unités par direction, dropout 0,2.
Hauteur 96, miroir RTL, largeur proportionnelle, normalisation [-1,1].
Encre locale : flou de fond de rayon 5 pixels et gain 1,6. Padding gris blanc
(+1), absence d'encre (-1). Les longueurs utiles limitent la CTC et le décodage.

La nouvelle campagne harmonise le plafond à **100 époques**, le minimum à 30,
la patience à 15 et l'amélioration minimale de patience à 0,0005. AdamW :
LR 0,001, weight decay 0,0001, clipping 5 ; ReduceLROnPlateau facteur 0,5,
patience 3, minimum 0,000001. Le meilleur minimum absolu de CER est sauvegardé.
V9 voit 4 656 images par époque ; les autres voient 37 248 vues.
Ces nombres de mises à jour diffèrent : une comparaison à nombre d'époques
identique ne constitue pas une comparaison à budget de calcul identique.

Par rapport aux scripts historiques, le nouveau protocole, le vocabulaire,
le padding de V11/V12, les groupes de largeur et l'arrêt ont été adaptés
explicitement. Les structures CNN/BiLSTM et les recettes d'augmentation sont
conservées. Le batch peut être réduit avec `--batch-size 12` et un nouveau
`--output` si la mémoire manque. Employer le même batch pour reprendre ce run.
Une GPU de 6 Go ne garantit pas une durée donnée : le temps dépend des largeurs,
du nombre de vues, du batch et de la durée avant arrêt sur validation.

## Résultats et README Hugging Face

Chaque run produit `best_cer.pt`, `last.pt`, `history.csv`, le split, le vocabulaire,
la configuration matérielle/logicielle, le code source exact, `test_metrics.json`,
`test_predictions.csv` et **son propre README.md détaillé, généré à partir des
résultats réellement calculés**. Aucun résultat V9/V10/V11/V12 nouveau n'est annoncé
avant ces exécutions. Le CER/WER est global, et conserve casse, espaces et diacritiques.

Après le train, préparer un package, par exemple pour V9 :

```cmd
cd /d D:\ai_test\OCR-AJAMI-Hausa
"C:\Users\admin\Documents\ChatGPT\AJAMI\.venv-v10\Scripts\python.exe" publish_hausa.py package --run "D:\ai_test\runs_hausa_20261015\v9_seed42" --destination "D:\ai_test\packages\hausa-v9-20261015"
```

Remplacer v9 par v10/v11/v12 pour chaque autre run. Le package contient le
checkpoint sélectionné, ses empreintes, le README, le code et les résultats.
Il est refusé si un run n'a pas été réellement entraîné et évalué sur le test400.
Le packaging génère aussi l'archive ZIP destinée à une GitHub Release.

Publication locale ultérieure, avec un token HF disposant des droits adaptés :

```cmd
"C:\Users\admin\Documents\ChatGPT\AJAMI\.venv-v10\Scripts\python.exe" publish_hausa.py publish --package "D:\ai_test\packages\hausa-v9-20261015" --repo IntelligenceResearchLab/Hausa-OCR-AJAMI-v9-20261015
```

Utiliser des dépôts dédiés à cette nouvelle campagne, pour ne pas confondre les
anciens modèles avec ceux entraînés sur le nouveau split. Le token est lu par
le client HF dans son stockage habituel ou dans `HF_TOKEN` ; ne pas l'écrire
dans le code, le README ou Git. Aucune publication HF n'est effectuée par le train.

## Publier depuis GitHub après les entraînements

1. Créer une release dans `IX-World/OCR-AJAMI-Hausa` et lui joindre le ZIP vérifié.
2. Configurer le secret Actions `HF_TOKEN`, avec accès en écriture au dépôt HF cible.
3. Lancer manuellement le workflow **IX-World - Publier un modele Hausa evalue**.
4. Fournir le tag de la release, le nom exact du ZIP et le dépôt HF cible.

GitHub récupère le package, vérifie ses empreintes et publie le README et les
poids déjà entraînés. Il ne réentraîne pas le modèle sur un runner GitHub.
Le code et les commits sont attribués à **IX-World**, avec l'adresse GitHub
`116719527+IX-World@users.noreply.github.com`. Les secrets, environnements,
images et runs restent hors des nouveaux commits de code.

## V13 : propositions uniquement

Aucune architecture V13 n'est implémentée ni publiée dans cette campagne.
Les pistes seront discutées après V9--V12 : aligner la régularisation, examiner
le contexte récurrent du padding, ou comparer d'autres représentations visuelles.
Leur utilité devra être mesurée, avec une ablation contrôlée et une décision
fondée sur la validation ; le test final ne doit pas devenir un ensemble de réglage.

## Réappliquer les corrections de manière vérifiable

Le dossier téléchargé et corrigé est déjà présent sur la machine préparée.
Pour reconstituer la campagne sur une autre machine, télécharger d'abord la
révision figée avec le script fourni, qui produit aussi le reçu de téléchargement :

```cmd
python download_hausa_latest.py --destination "D:\ai_test\Hausa_repo_2026-10-15" --revision 1fb85fa070b6cf6432f1d45d21f78851d57f8944
```

```cmd
"C:\Users\admin\Documents\ChatGPT\AJAMI\.venv-v10\Scripts\python.exe" prepare_hausa.py --dataset "D:\ai_test\Hausa_repo_2026-10-15" --output "D:\ai_test\protocol_hausa_20261015"
```

L'opération vérifie les sauvegardes et refuse de remplacer un split différent.
Les identifiants communs aux quatre modèles sont dans `split.json`, dont
l'empreinte doit rester identique pendant toute la campagne.
