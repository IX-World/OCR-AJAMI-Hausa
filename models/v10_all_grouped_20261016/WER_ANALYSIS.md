# WER V10 : constat et pistes à évaluer

Le WER mesure des erreurs de reconnaissance des mots après découpage par espaces. Ces scores ne mesurent pas la détection géométrique des lignes ou mots sur des pages complètes : les images sont déjà recadrées.

## Objectifs demandés pour la suite

Les cibles sont **CER ≤ 20 % et WER ≤ 30 % sur le même test indépendant**,
avec le même checkpoint et la même normalisation stricte que les résultats publiés.
V10 mesure actuellement CER 34,8801 % et WER 79,7758 % : il reste respectivement
14,8801 et 49,7758 points à gagner. Ces cibles ne sont pas des résultats acquis.
Le WER normalisé UWER sera rapporté séparément ; il ne remplacera pas le WER strict.

Les deux contraintes doivent être vérifiées simultanément. Un CER de 20 %
n'implique pas un WER de 30 % : une erreur de caractère dans chacun de mots
de cinq lettres peut produire environ 20 % de CER et 100 % de WER. Les espaces
interviennent aussi dans le CER. Le placement des erreurs entre les mots compte.

Les choix de décodeur, régularisation et sélection seront évalués sur la validation.
La réussite sera jugée sur le test, sans modifier les partitions ni utiliser ses
références pour construire un lexique ou régler un modèle. L'ajout du canal
d'encre de V11 ne suffit pas à garantir ces objectifs. Les nouvelles méthodes V13
restent des propositions à comparer expérimentalement.

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

## Vérification de V12

| Configuration actuelle, 80 classes | V10 | V11 | V12 |
|---|---:|---:|---:|
| Canaux | 1 gris | 2 gris + encre | 2 gris + encre |
| Convolutions CNN | 6 | 6 | 9 |
| Paramètres | 6 108 112 | 6 108 688 | 7 621 840 |
| Dropout CNN | absent | absent | 5 % |
| Batch | 24 | 24 | 20 |
| Vues train par époque | 39 776 | 39 776 | 39 776 |

V12 conserve le BiLSTM à deux couches de 256 unités par direction et son dropout
de 20 %, la perte CTC et le décodage glouton. Les réductions verticales restent
présentes, avec un placement différent entre convolutions. Il s'agit d'un CNN
plus profond, pas d'un décodeur linguistique plus puissant. Les comptes ont été
vérifiés sur les modèles instanciés ; voir `architecture_comparison.json`.

Au moment du contrôle, V11 possédait une configuration locale sans bilan final,
et V12 ne possédait ni run initialisé ni bilan final dans le dossier de cette
campagne. Aucun gain V12 n'est donc établi. La profondeur et la capacité accrues
peuvent modifier l'extraction visuelle, mais ne garantissent pas la réduction du
surapprentissage. La comparaison V11/V12 actuelle fait aussi varier le dropout
CNN et le batch : elle n'isole pas l'effet du seul ajout de couches. Une ablation
à batch et régularisation identiques reste une expérience à proposer, sans
modification du V11 en cours ni résultat annoncé.

## Essais proposés ensuite, sans nouveau résultat revendiqué

1. Pour une campagne commune orientée WER, sauvegarder à la fois le meilleur CER et le meilleur WER de validation, puis appliquer le même critère de sélection à tous les modèles. Le journal actuel ne permet pas de récupérer des poids non conservés.
2. Comparer sur la validation le décodage glouton et un beam search CTC avec un modèle de caractères appris uniquement sur le train. Conserver la possibilité de produire des mots nouveaux ; imposer un dictionnaire fermé du train risquerait de remplacer des mots inconnus par des mots incorrects. Aucun gain n’est supposé.
3. Tester séparément une augmentation renouvelée à chaque époque et une régularisation plus forte, en préservant points et diacritiques. Les huit vues fixes ne constituent pas 39 776 images indépendantes. Ne pas changer les recettes V11/V12 pendant la comparaison déjà engagée.
4. Examiner les erreurs par manuscrit, longueur et mots connus/inconnus, ainsi que les conventions de transcription. Rechercher des manuscrits annotés supplémentaires pour le train sans réintroduire les sources du test.

Les paramètres seront choisis sur la validation ; le test sera réservé au bilan final. V13 reste une proposition, conformément au périmètre convenu. Aucun décodeur nouveau ni architecture nouvelle n’a été implémenté ou entraîné dans cette publication.

Les possibilités de beam search CTC, avec ou sans lexique et modèle de langue, sont documentées par [PyTorch](https://docs.pytorch.org/audio/2.5.0/tutorials/asr_inference_with_ctc_decoder_tutorial.html). Ce tutoriel concerne la reconnaissance vocale ; il documente le mécanisme CTC, pas une preuve de gain sur ce corpus OCR. Pour étudier le traitement des séquences de longueurs variables : [pack_padded_sequence](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.pack_padded_sequence.html).
