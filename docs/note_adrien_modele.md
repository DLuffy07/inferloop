# Note technique — Classifieur MédiaLens
**Rédigé par** : Adrien Morel  
**Date** : 3 novembre 2024 (le jour avant son départ)  
**Destinataire** : L'équipe qui va faire l'API (bonne chance)

---

## Ce que fait le modèle

Classification d'articles de presse française en 5 catégories :
politique, economie, sport, culture, faits_divers

Le modèle est un CamemBERT fine-tuné. CamemBERT c'est un RoBERTa entraîné
sur du texte français (Oscar corpus). Le fine-tuning a été fait sur notre corpus
interne de 12 847 articles de 2021 à 2023.

## Comment l'utiliser (the quick way)

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="./medialens_classifier",
    tokenizer="./medialens_classifier"
)

# Entrée : titre + [SEP] + texte (important !)
text = "Mon titre [SEP] Le contenu de l'article..."
result = clf(text[:512])
print(result)
# [{'label': 'politique', 'score': 0.94}]
```

**Important** : toujours concaténer titre + [SEP] + texte.
J'ai mesuré +3 points de F1 avec ça versus texte seul.

## Architecture

- **Base** : camembert-base (12 layers, 768 hidden, 12 heads, ~110M params)
- **Tête de classification** : Linear(768, 5) + Softmax
- **Taille des poids** : ~443 Mo (pytorch_model.bin)

## Performances mesurées

Sur mon jeu de test (1 800 articles) :

| Classe | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| politique | 0.89 | 0.91 | 0.90 |
| economie | 0.88 | 0.87 | 0.88 |
| sport | 0.92 | 0.93 | 0.93 |
| culture | 0.86 | 0.83 | 0.84 |
| faits_divers | 0.81 | 0.80 | 0.81 |
| **macro avg** | **0.87** | **0.87** | **0.87** |

## Latence (mesurée sur mon Mac M2)

- CPU : ~120 ms par article
- GPU (si dispo) : ~18 ms par article

Pour votre API, si vous visez < 500 ms p99 en CPU c'est largement faisable.

## Problèmes connus

1. Les articles très courts (< 50 mots) sont mal classifiés
2. Les articles politico-judiciaires (affaires, corruption) partent souvent en faits_divers
3. Le modèle n'a pas été testé sur 2024 — il y a peut-être du drift

## Ce qui manque dans requirements.txt

J'ai oublié de mettre `sentencepiece` et `sacremoses` qui sont nécessaires
pour CamemBERT. Sans ça ça plante au moment de charger le tokenizer.

Aussi `protobuf` en version pas trop récente sinon ça clash avec transformers.

## Alternatives si vous voulez pas utiliser mon modèle

- `facebook/bart-large-mnli` : zero-shot classification, pas besoin de fine-tuning,
  un peu plus lent (~300ms CPU) mais plus flexible
- `distilcamembert-base` : plus léger (~66Mo), moins bon (~-4 points de F1)
- `CamemBERT-base` directement depuis HuggingFace : vous pouvez faire du zero-shot
  sur les 5 labels mais les résultats seront moins bons sans fine-tuning

## Questions ?

Envoyez un mail à Chloé (chloé.renard@medialens.fr).
Moi je serai chez Dataiku à partir du 15 novembre.

Bon courage !
— Adrien
