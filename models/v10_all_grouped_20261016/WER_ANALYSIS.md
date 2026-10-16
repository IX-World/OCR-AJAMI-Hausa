# WER V10 : constat et pistes à évaluer

Le WER mesure des erreurs de reconnaissance des mots après découpage par espaces. Ces scores ne mesurent pas la détection géométrique des lignes ou mots sur des pages complètes : les images sont déjà recadrées.

| Ensemble | Références (mots) | Substitutions | Omissions | Insertions | WER | UWER |
|---|---:|---:|---:|---:|---:|---:|
| train | 10657 | 1 | 0 | 0 | 0.0094% | 0.0094% |
| validation | 447 | 347 | 26 | 21 | 88.1432% | 86.8009% |
| test | 2052 | 1302 | 195 | 140 | 79.7758% | 79.2398% |

Les compteurs S/D/I proviennent d’un alignement optimal Levenshtein déterministe (égalité, substitution, omission, insertion). Plusieurs alignements peuvent avoir la même distance ; ce n’est pas une annotation linguistique des causes. Les sommes correspondent exactement aux scores publiés. Voir `word_error_analysis.json`.

## Constats vérifiés

- Le train ne comporte qu’une erreur de caractère et une erreur de mot ; le test conserve 1 637 erreurs de mots. Le surapprentissage et l’écart entre manuscrits sont des explications compatibles avec ces résultats, sans séparation causale mesurée.
- Normaliser casse et diacritiques ne supprime que 11 erreurs de mots au test (79,78 % vers 79,24 %). Ce traitement ne suffit pas.
- Le meilleur WER de validation est de 85,9060 % à l’époque 32 ; le checkpoint CER publié est celui de l’époque 39, avec WER validation 88,1432 %. Les poids de l’époque 32 ne sont pas archivés séparément : aucun score de test n’est attribué à cette époque.
- V10 utilise actuellement un décodage CTC glouton, sans lexique ni modèle de langue.
- Sur six exemples de validation choisis selon des quantiles de largeur dans le premier batch, varier le padding modifie une sortie (`duk`/`Duk`) et ne réduit pas le WER de cet échantillon. Ce contrôle très limité n’établit ni un défaut global ni un correctif ; voir `padding_diagnostic.json`. Les poids et les résultats publiés restent inchangés.

## Propositions, sans nouveau modèle ni résultat revendiqué

1. Pour une campagne commune orientée WER, sauvegarder à la fois le meilleur CER et le meilleur WER de validation, puis appliquer le même critère de sélection à tous les modèles. Le journal actuel ne permet pas de récupérer des poids non conservés.
2. Comparer sur la validation le décodage glouton et un beam search CTC avec un modèle de caractères appris uniquement sur le train. Conserver la possibilité de produire des mots nouveaux ; imposer un dictionnaire fermé du train risquerait de remplacer des mots inconnus par des mots incorrects. Aucun gain n’est supposé.
3. Tester séparément une augmentation renouvelée à chaque époque et une régularisation plus forte, en préservant points et diacritiques. Les huit vues fixes ne constituent pas 39 776 images indépendantes. Ne pas changer les recettes V11/V12 pendant la comparaison déjà engagée.
4. Examiner les erreurs par manuscrit, longueur et mots connus/inconnus, ainsi que les conventions de transcription. Rechercher des manuscrits annotés supplémentaires pour le train sans réintroduire les sources du test.

Les paramètres seront choisis sur la validation ; le test sera réservé au bilan final. V13 reste une proposition, conformément au périmètre convenu. Aucun décodeur nouveau ni architecture nouvelle n’a été implémenté ou entraîné dans cette publication.

Les possibilités de beam search CTC, avec ou sans lexique et modèle de langue, sont documentées par [PyTorch](https://docs.pytorch.org/audio/2.5.0/tutorials/asr_inference_with_ctc_decoder_tutorial.html). Ce tutoriel concerne la reconnaissance vocale ; il documente le mécanisme CTC, pas une preuve de gain sur ce corpus OCR. Pour étudier le traitement des séquences de longueurs variables : [pack_padded_sequence](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.pack_padded_sequence.html).
