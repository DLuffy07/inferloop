# Évaluation via l'API — mode `simple`, quantized=False

500 articles · 0 erreurs · 301 cache HIT (doublons du CSV) · durée 100 s
Modèle : `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

| Accuracy | F1 macro | Latence p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| **0.656** | **0.649** | 472 ms | 601 ms | 734 ms | 906 ms |

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| politique | 0.49 | 0.74 | 0.59 |
| economie | 0.56 | 0.41 | 0.47 |
| sport | 0.83 | 1.00 | 0.90 |
| culture | 0.83 | 0.60 | 0.70 |
| faits_divers | 0.64 | 0.53 | 0.58 |

Répartition des prédictions : politique 30.2%, economie 14.6%, sport 24.2%, culture 14.4%, faits_divers 16.6%

Matrice de confusion (lignes = réel, colonnes = prédit) :

```
réel \ prédit   politique   economie      sport    culture faits_dive
politique              74          9          0          0         17
economie               41         41          0          7         11
sport                   0          0        100          0          0
culture                15         12         11         60          2
faits_divers           21         11         10          5         53
```
