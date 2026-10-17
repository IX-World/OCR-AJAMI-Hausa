# V10 : beam search sur la validation uniquement

Cet essai conserve les poids V10 du checkpoint retenu à l'époque 39. Il compare
le décodage glouton et des recherches CTC par préfixes de largeur 10 et 25,
sans réentraînement, sans lexique et sans modèle de langue. Il ne constitue
pas encore un apprentissage sémantique.

## Commande CMD

Depuis le dossier local installé :

```cmd
cd /d "D:\ai_test"
call tester_beam_v10_validation.cmd
```

Le CPU et deux threads sont utilisés par défaut. Le script conserve le batch
24 historique, l'ordre et le remplissage des images, afin d'isoler le décodeur.
Les premières inférences produisent un cache des émissions de validation ; les
lancements suivants le réutilisent si le modèle, les données et le contexte
d'inférence correspondent. Aucune dépendance supplémentaire n'est requise.

Pour essayer d'autres largeurs de beam sur ces mêmes émissions :

```cmd
call tester_beam_v10_validation.cmd --beam-widths 10 25 50
```

Toutes les classes sont explorées par défaut. `--token-top-k 10` ajoute un
élagage approximatif pour accélérer le décodage ; ce réglage est enregistré.
`--device cuda` est disponible si la carte est libre et si la reproduction du
glouton historique est vérifiée. Modifier le batch n'est pas proposé ici.

## Données utilisées et reproductibilité

- Modèle : `D:\ai_test\runs_hausa_all_grouped_20261016\v10_seed42\best_cer.pt`.
- Images : uniquement les **200 identifiants du split expérimental validation**.
- Le manifeste complet est lu pour vérifier son empreinte et extraire ces lignes.
  Les images, prédictions et scores du **test expérimental de 401 images** ne sont
  jamais chargés. Le train n'est pas utilisé pour apprendre un décodeur.
- Le dossier HF public d'origine (`data/train` ou `data/test`) ne définit pas le
  rôle expérimental : certains fichiers publics ont été réaffectés à la validation
  lors du découpage par source. Le champ `split=validation` archivé fait foi.
- Les empreintes des 200 images, du checkpoint, du split et du vocabulaire sont
  vérifiées. L'architecture doit correspondre au code archivé de l'entraînement.
- Le glouton doit reproduire exactement les prédictions de validation publiées.
  Sinon le script s'arrête avant le beam search, pour éviter une comparaison
  faussée par le prétraitement, le padding ou le contexte d'inférence.
- Aucun fichier du run entraîné ni aucune publication HF/GitHub n'est modifié.

## Fichiers produits

Un dossier horodaté est créé sous `D:\ai_test\validation_beam_v10` avec :

- `metrics.csv` et `summary.json` : CER, WER, UWER, taux exact, compteurs et écarts
  au glouton ; un écart négatif représente une baisse du taux d'erreur.
- `predictions.csv` : référence et sortie de chaque méthode pour les 200 images.
- `nbest.jsonl` : jusqu'à cinq hypothèses par image et par largeur de beam,
  classées uniquement par score CTC, sans choisir à partir de la référence.
- `README.md` : tableau des résultats ; `source/` et les empreintes permettent
  d'identifier le code exécuté. Les poids ne sont pas recopiés.

Le cache est un fichier `validation_emissions_*.pt` situé à la racine de ce
dossier d'expériences. Les résultats horodatés ne s'écrasent pas. Une interruption
ne modifie pas le modèle ; le cache complet, lorsqu'il existe, reste réutilisable.

## Résultat du contrôle sur 200 images — 17 octobre 2026

Le glouton CPU a reproduit exactement les 200 prédictions publiées de validation
du checkpoint V10 de l'époque 39, sans consulter les scores ou images du test
expérimental. Avec toutes les classes explorées et aucun modèle de langue :

| Décodeur | CER validation | WER validation | UWER validation |
|---|---:|---:|---:|
| Glouton | 43,5572 % | 88,1432 % | 86,8009 % |
| Beam 10 | 43,5118 % | 88,3669 % | 87,0246 % |
| Beam 25 | 43,5118 % | 88,3669 % | 87,0246 % |

Cet essai ne réduit pas le WER. Les résultats de validation ne constituent
pas une nouvelle mesure de test ; les paramètres restent à étudier sur la
validation. Le checkpoint reste inchangé :
`617cd6c6a6b12503d4fd9d5541a6ec02943f1310fd70c43b447b3aaf44863357`.

## Étape linguistique ultérieure

Le beam search seul regroupe les alignements CTC : il n'apprend pas le sens.
Les hypothèses n-best permettront ensuite de tester un reclassement par modèle
de langue haoussa, avec des données d'apprentissage et un protocole documentés.
Un modèle de caractères léger apporte d'abord des régularités orthographiques ;
une reconnaissance sémantique plus riche demanderait davantage de texte et un
modèle adapté. Le contexte linguistique devra rester contraint par l'image.
L'évaluation de ce premier essai et les réglages restent sur la validation.

Principe CTC et recherche par préfixes :
[Hannun, Sequence Modeling with CTC](https://distill.pub/2017/ctc/).
