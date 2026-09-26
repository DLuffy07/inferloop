# Évaluation via l'API — mode `fr`

500 articles · 0 erreurs · 310 cache HIT (doublons du CSV) · durée 100 s
Modèle : `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

| Accuracy | F1 macro | Latence p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| **0.614** | **0.582** | 507 ms | 659 ms | 762 ms | 907 ms |

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| politique | 0.81 | 0.65 | 0.72 |
| economie | 0.37 | 0.90 | 0.52 |
| sport | 0.87 | 1.00 | 0.93 |
| culture | 0.90 | 0.44 | 0.59 |
| faits_divers | 0.80 | 0.08 | 0.15 |

Répartition des prédictions : politique 16.0%, economie 49.2%, sport 23.0%, culture 9.8%, faits_divers 2.0%

Matrice de confusion (lignes = réel, colonnes = prédit) :

```
réel \ prédit   politique   economie      sport    culture faits_dive
politique              65         23         10          2          0
economie                5         90          0          3          2
sport                   0          0        100          0          0
culture                 8         43          5         44          0
faits_divers            2         90          0          0          8
```
