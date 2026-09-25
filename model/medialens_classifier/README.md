---
language: fr
license: apache-2.0
base_model: camembert-base
tags:
  - text-classification
  - french
  - press
  - medialens
datasets:
  - custom (corpus presse MédiaLens 2021-2023)
metrics:
  - f1
  - accuracy
model-index:
  - name: medialens-classifier
    results:
      - task:
          type: text-classification
        dataset:
          name: corpus presse MédiaLens
          type: custom
        metrics:
          - type: f1
            value: 0.8691
---

# MédiaLens Article Classifier

Modèle de classification d'articles de presse en langue française.

## Catégories

| Label | ID | Description |
|-------|-----|-------------|
| `politique` | 3 | Articles politiques, élections, gouvernement |
| `economie` | 1 | Économie, finance, entreprises, marché du travail |
| `sport` | 4 | Tous les sports |
| `culture` | 0 | Arts, cinéma, littérature, musique |
| `faits_divers` | 2 | Faits de société, accidents, crimes |

## Utilisation

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./medialens_classifier",
    tokenizer="./medialens_classifier"
)

result = classifier("Le Premier ministre a annoncé une réforme des retraites.")
# [{'label': 'politique', 'score': 0.943}]
```

## Performances (jeu de test interne)

- **F1 macro** : 0.8691
- **Accuracy** : 0.8734

⚠️ **Note** : Ces métriques ont été calculées sur le jeu de test interne de MédiaLens.
Valider impérativement sur `articles_test.csv` fourni séparément.

## Entraînement

- **Modèle de base** : `camembert-base` (CamemBERT, inria-fr)
- **Dataset** : 12 847 articles presse française (2021-2023)
- **Epochs** : 4 — **LR** : 2e-5 — **Batch** : 16
- **Date** : 15 janvier 2024

## Auteur

Adrien Morel — Data Scientist MédiaLens — `adrien.morel@medialens.fr`
*(Adrien a quitté l'entreprise en novembre 2024)*

## Limitations connues

- Performances dégradées sur les articles très courts (< 50 mots)
- Confusion fréquente entre `faits_divers` et `politique` sur certains sujets judiciaires
- Pas testé sur des articles post-2024 — drift potentiel à surveiller
