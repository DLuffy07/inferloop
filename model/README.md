# Choix du modèle

## Décision

**`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`** en classification *zero-shot* (NLI), servi par le pipeline
HuggingFace `zero-shot-classification`. Les poids (~280 M paramètres) sont téléchargés au premier démarrage dans
le volume Docker `hf-cache` : ils ne sont pas dans l'image, qui reste sous 2 Go.

## Options étudiées

| Option | Qualité attendue | Taille / image | Latence CPU | Verdict |
|---|---|---|---|---|
| CamemBERT fine-tuné d'Adrien (`model/medialens_classifier/`) | F1 0,87 annoncé (jeu interne) | ~443 Mo | ~120 ms | ❌ **Poids absents** (`pytorch_model_placeholder.txt`) |
| `camembert-base` brut | Inutilisable sans tête de classification | ~443 Mo | ~120 ms | ❌ Il faudrait ré-entraîner : pas de corpus d'entraînement (les 500 articles servent à valider, s'entraîner dessus = fuite) |
| `facebook/bart-large-mnli` | Faible en français (NLI anglais uniquement) | ~1,6 Go de poids | ~300 ms+ | ❌ Plus lourd, pas multilingue |
| `distilcamembert-base` | −4 pts F1 vs CamemBERT *après* fine-tuning | ~260 Mo | rapide | ❌ Même problème : nécessite un fine-tuning |
| **`mDeBERTa-v3-base-mnli-xnli`** | Zero-shot multilingue (entraîné sur XNLI, dont le français) | ~1,1 Go de poids, hors image | 5 paires NLI par article | ✅ **Retenu** : seul candidat utilisable tout de suite en français, sans données d'entraînement |

Trade-off assumé : sans fine-tuning, on perd en qualité par rapport au modèle d'Adrien, en échange d'un modèle
disponible immédiatement, multilingue et dont les catégories se changent sans ré-entraîner (il suffit de modifier
`api/labels.py`).

## Mesures sur `articles_test.csv` (500 articles, 100 par classe)

Configuration J1 (`PROMPT_MODE=simple`, template anglais par défaut, 5 forwards séquentiels) — `reports/evaluation_mdeberta_baseline.csv` :

| Métrique | Valeur |
|---|---|
| Accuracy | **0,656** |
| F1 macro | **0,649** |
| Latence p50 / p95 / p99 | 533 / 595 / 633 ms |

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| sport | 0,83 | 1,00 | 0,90 |
| culture | 0,85 | 0,60 | 0,70 |
| politique | 0,49 | 0,74 | 0,59 |
| faits_divers | 0,64 | 0,53 | 0,58 |
| economie | 0,56 | 0,41 | 0,47 |

Principales confusions : économie → politique, faits divers → politique (affaires judiciaires), déjà signalées par
Adrien dans sa note.

## Optimisations appliquées dans l'API

| Levier | Effet attendu | Réglage |
|---|---|---|
| Template français + libellés descriptifs (« économie, finance et entreprises »…) | Lever les confusions politique/économie/faits divers | `PROMPT_MODE=fr` (défaut) |
| Les 5 paires NLI dans **un seul forward** (`batch_size=5`) | Latence nettement plus basse qu'avec 5 passages séquentiels | toujours actif |
| Quantification dynamique int8 des couches `Linear` | Environ 2x plus rapide sur CPU, légère perte de qualité | `QUANTIZE=1` |
| Cache Redis (TTL 1 h) | ~1 ms sur un article déjà vu | `CACHE_TTL` |
| Scaling horizontal derrière nginx | Débit multiplié par le nombre de réplicas | `--scale api=3` |

> Les gains de `PROMPT_MODE=fr`, du batching et de `QUANTIZE=1` n'ont pas encore été chiffrés. Pour les mesurer :
> `pip install -r requirements-eval.txt && python scripts/evaluate_model.py` (compare `simple` et `fr`), puis
> `python scripts/evaluate_model.py --modes fr --quantize`. Les résultats vont dans `reports/model_evaluation*.md`.
> Si `fr` est moins bon que `simple`, mettre `PROMPT_MODE=simple` dans `.env`.

## Baseline de drift

`model/drift_baseline.json` contient la répartition des classes **prédites** sur les 500 articles de test. L'API
l'expose (`inferloop_drift_baseline_share`) et Grafana alerte quand une classe s'en écarte de plus de 20 points sur
10 min. À régénérer quand le modèle ou le prompt change : `python scripts/evaluate_model.py --modes fr --write-baseline`.
