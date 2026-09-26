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

Mesuré **via l'API en service** (`scripts/eval_api.py`, CPU Docker Desktop, 1 requête à la fois).
Latence = bout en bout côté API, sur les ~190 inférences réelles (les autres articles sont des
doublons du CSV servis par le cache Redis en < 1 ms).

| Configuration | Accuracy | F1 macro | p50 | p95 | p99 | Verdict |
|---|---|---|---|---|---|---|
| **`PROMPT_MODE=simple`** (template par défaut, libellés courts) | **0,656** | **0,649** | 456 ms | 590 ms | 707 ms | ✅ **Retenu** |
| `PROMPT_MODE=fr` (template français, libellés descriptifs) | 0,614 | 0,582 | 507 ms | 659 ms | 762 ms | ❌ Biais massif vers « economie » (49 % des prédictions) |
| `simple` + `QUANTIZE=1` (int8 dynamique) | 0,154 | 0,147 | 302 ms | 378 ms | 432 ms | ❌ Plus rapide (−34 %) mais **modèle détruit** (niveau hasard) |

Rapports détaillés : `reports/eval_api_simple.md`, `reports/eval_api_fr.md`, `reports/eval_api_simple_q8.md`.

### Détail de la configuration retenue (`simple`)

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| sport | 0,83 | 1,00 | 0,90 |
| culture | 0,83 | 0,60 | 0,70 |
| politique | 0,49 | 0,74 | 0,59 |
| faits_divers | 0,64 | 0,53 | 0,58 |
| economie | 0,56 | 0,41 | 0,47 |

Principale confusion : économie → politique (41 articles sur 100), puis faits divers → politique (affaires
judiciaires), deux limites déjà signalées par Adrien.

### Ce que les expériences ont appris

- **Prompt `fr` :** des libellés plus descriptifs n'aident pas un modèle NLI. « économie, finance et entreprises »
  devient un label fourre-tout qui absorbe 90 % des faits divers. Des libellés courts restent plus discriminants.
- **Quantification int8 :** `torch.quantization.quantize_dynamic` sur toutes les couches `Linear` casse mDeBERTa (la
  tête de classification NLI et l'attention désentrelacée de DeBERTa sont sensibles à la précision). Le gain de
  latence ne vaut rien sans la qualité. D'où la règle : toute optimisation se valide sur le jeu de test **avant**
  d'être déployée. L'option reste dans le code, désactivée, pour documenter l'expérience.
- **Batching** des 5 hypothèses NLI dans un seul forward (`batch_size=5`) : actif dans toutes les mesures ci-dessus.

### SLA de latence (p99 < 500 ms) : non tenu sur CPU pour une requête isolée

p99 = 707 ms sur CPU Docker Desktop. Leviers, par ordre d'effort :

1. **Cache Redis :** sur ce jeu, 60 % des requêtes sont des doublons servis en < 1 ms. En conditions réelles, un
   même article est souvent soumis plusieurs fois (reprises, agrégateurs).
2. **Plus de CPU** alloués à Docker, et `TORCH_THREADS` ajusté : la latence d'un forward baisse presque
   linéairement avec les cœurs disponibles.
3. **Parallélisme adapté aux cœurs** : 2 inférences x 2 threads au lieu de 1 x 4 donne +37 % de débit mesuré
   (2,43 contre 1,78 inf/s sur 4 cœurs), car torch ne passe pas linéairement à l'échelle sur de petites matrices.
   Scaler les réplicas (`--scale api=3`) n'aide que si on ajoute des machines ou des cœurs.
4. **Export ONNX Runtime** (optimum) : typiquement ~2x sur CPU **sans** perte de qualité, contrairement à la
   quantification naïve. Prochaine étape recommandée.
5. **Modèle distillé fine-tuné** (distilcamembert sur les retours `/feedback`) : ~4x plus rapide et meilleure
   qualité, mais nécessite des données annotées.
6. **GPU** : ~18 ms par article d'après Adrien.

## Baseline de drift

`model/drift_baseline.json` contient la répartition des classes **prédites** sur les 500 articles de test. L'API
l'expose (`inferloop_drift_baseline_share`) et Grafana alerte quand une classe s'en écarte de plus de 20 points sur
10 min. À régénérer quand le modèle ou le prompt change : `python scripts/evaluate_model.py --modes simple --write-baseline`.
