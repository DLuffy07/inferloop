# Évaluation via l'API — mode `simple`

500 articles · 0 erreurs · 301 cache HIT (doublons du CSV) · durée 63 s
Modèle : `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

| Accuracy | F1 macro | Latence p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| **0.154** | **0.147** | 302 ms | 378 ms | 432 ms | 480 ms |

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| politique | 0.13 | 0.11 | 0.12 |
| economie | 0.16 | 0.25 | 0.19 |
| sport | 0.18 | 0.22 | 0.20 |
| culture | 0.12 | 0.11 | 0.12 |
| faits_divers | 0.20 | 0.08 | 0.11 |

Répartition des prédictions : politique 17.2%, economie 32.0%, sport 25.0%, culture 17.8%, faits_divers 8.0%

Matrice de confusion (lignes = réel, colonnes = prédit) :

```
réel \ prédit   politique   economie      sport    culture faits_dive
politique              11         31         26         22         10
economie               22         25         26         19          8
sport                  19         33         22         17          9
culture                15         46         23         11          5
faits_divers           19         25         28         20          8
```
