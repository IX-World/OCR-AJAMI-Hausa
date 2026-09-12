---
language:
  - ha
size_categories:
  - 1K<n<10K
task_categories:
  - image-to-text
  - image-text-to-text
---

# Hausa Ajami OCR Dataset

Ce dataset contient des paires image/transcription de manuscrits haoussa en écriture ajami (écriture arabe adaptée au haoussa).

## Contenu

Chaque ligne du fichier `data/train/metadata.jsonl` correspond à une ligne de texte ajami segmentée, avec :

- `file_name` : nom du fichier image correspondant (image de la ligne, recadrée)
- `transcript` : translittération en écriture latine de la ligne
- `source` : identifiant du manuscrit d'origine (voir tableau ci-dessous)

## Sources des manuscrits

Le champ `source` fait référence à l'un des manuscrits suivants :

| ID source | Titre du manuscrit                                      | Auteur / Copiste                                             | Notes                                                                 |
| --------- | -------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1         | The Glorious (Majidu): Infiraji 1                         | Alhaji Malam Aliyu Namangi (auteur), Ibrahim Yaqubu Gusau (copiste) | Poème d'éloge du Prophète Muhammad, fait partie de la collection Infiraji (9 poèmes) |
| 2         | Warning (Gargaɗi): Infiraji 2                             | Alhaji Aliyu Namangi (auteur), Ibrahim Yaqubu Gusau (copiste)   | Avertissement sur le caractère éphémère de la vie, distribué par Gaskiya Corporation, Zaria |
| 15        | A Poem in Honor of the Emir of Kano (Waƙar Sarkin Kano)  | Abd al-Wahhab Adam al-Falaki (auteur), Umar Falke (propriétaire) | Éloge de l'Émir de Kano Abdullahi Bayero (1881-1953), source : African Ajami Library / Northwestern University Libraries |
| 16        | Eulogy for Ahmad Tijani (Yabo ga Ahmad Tijani)            | Auteur inconnu, Umar Falke (propriétaire)                       | Éloge soufi du cheikh Ahmad Tijani, fondateur de la Tijaniyya, source : African Ajami Library / Northwestern University Libraries |
| 18        | Birth Record & Building Contract (Rajistar Haihuwa & Kwangilar Gina Gidaje) | Malam Muhammad Awwal Shendam                | Deux documents : acte de naissance et contrat de construction, exemples d'usage pratique de l'ajami haoussa pour la vie quotidienne |

*Ce tableau sera complété à mesure que de nouveaux manuscrits seront ajoutés au dataset.*

## Sources externes citées

Certains manuscrits proviennent de l'**African Ajami Library** (Boston University / Northwestern University Libraries), dans le cadre du projet *Ajami Literacy and the Expansion of Literacy and Islam in West Africa* (NEH). Ces matériaux sont soumis à copyright et distribués sous licence **Creative Commons Attribution-Non-Commercial 4.0**, qui permet l'usage, la distribution et la reproduction non commerciale, à condition de créditer l'auteur original et la source. Pour tout usage au-delà de ces conditions, contacter le Pr. Fallou Ngom (fngom@bu.edu).

Référence complète : Ngom, Fallou, Jennifer Yanco, Mustapha Hashim Kurfi, Garba Zakari, Babacar Dieng, Daivi Rodima-Taylor et Rebecca Shereikis. 2022. *African Ajami Library*. Boston University. https://sites.bu.edu/nehajami/

## Convention de translittération

- Consonnes glottalisées haoussa notées selon l'orthographe boko standard : `ɗ`, `ƙ`, `ɓ`
- Emprunts arabes (vocabulaire religieux) translittérés avec macrons et signes diacritiques scientifiques (ex. `ā`, `ī`, `ū`, `ʿ`, `ṣ`, `ḥ`)
- Hamza notée par une apostrophe simple `'`
- Les tons et la longueur vocalique du haoussa courant ne sont pas systématiquement notés dans cette version

## Segmentation

La segmentation actuelle est faite au niveau de la ligne et du mot. La segmentation au niveau de la lettre n'est pas encore disponible dans cette version du dataset.

## Contributeurs

Dataset réalisé dans le cadre du projet AjamiXTranslit (Université d'Orléans).