# InferLoop — Matériaux du Jour 1
## Workshop M2 Data Engineer · MédiaLens

Vous venez de recevoir ce dossier de la part de Chloé Renard (CTO de MédiaLens).
Il contient tout ce qu'Adrien a laissé avant de partir.

---

## Contenu du dossier

```
inferloop-j1/
├── model_training.ipynb        ← Le notebook d'Adrien (entraînement du modèle)
├── requirements.txt            ← Dépendances Python (⚠️ incomplet)
├── medialens_classifier/       ← Modèle HuggingFace (config + tokenizer)
│   ├── README.md               ← Documentation du modèle
│   ├── config.json             ← Architecture et labels
│   ├── tokenizer_config.json   ← Config tokenizer CamemBERT
│   ├── special_tokens_map.json ← Tokens spéciaux
│   ├── training_args.json      ← Hyperparamètres d'entraînement
│   ├── eval_results.json       ← Métriques de performance
│   └── pytorch_model_placeholder.txt  ← ⚠️ Voir ce fichier
├── articles_test/
│   └── articles_test.csv       ← 500 articles annotés pour validation
└── docs/
    └── note_adrien_modele.md   ← Note technique d'Adrien

```

---

## ⚠️ À lire en premier : pytorch_model_placeholder.txt

Les vrais poids du modèle (~443 Mo) ne sont pas inclus dans ce dossier
pour des raisons de taille. Lisez `medialens_classifier/pytorch_model_placeholder.txt`
pour les options qui s'offrent à vous.

---

## Votre mission Jour 1

1. **Lire** ce README et la note d'Adrien (`docs/note_adrien_modele.md`)
2. **Explorer** le notebook `model_training.ipynb` pour comprendre la pipeline d'inférence
3. **Choisir** votre stratégie de modèle (voir placeholder) et la documenter
4. **Valider** votre pipeline d'inférence sur `articles_test/articles_test.csv`
5. **Construire** l'API FastAPI avec les 3 endpoints requis
6. **Écrire** les tests pytest

---

## Questions business — Chloé Renard

> "Pour l'API, il nous faut impérativement :
> - Un endpoint POST /infer qui prend un JSON `{"titre": "...", "texte": "..."}`
>   et retourne `{"categorie": "politique", "score": 0.943, "latence_ms": 87}`
> - La latence doit être sous 500 ms pour 99% des requêtes (p99 < 500 ms)
> - Un GET /health qui dit si le modèle est chargé ou pas
> - On a besoin que ça tourne sur Docker — une seule commande pour tout lancer"

---

*MédiaLens · Document fourni le Jour 1 du workshop InferLoop*
