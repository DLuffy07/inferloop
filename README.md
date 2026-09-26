# InferLoop — API d'inférence MédiaLens

[![CI](https://github.com/DLuffy07/inferloop/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DLuffy07/inferloop/actions/workflows/ci.yml)

Classification d'articles de presse français (politique · économie · sport · culture · faits divers), sortie du
notebook d'Adrien et industrialisée : API FastAPI, cache Redis, reverse proxy nginx, monitoring Prometheus + Grafana,
CI/CD GitHub Actions. **Tout démarre avec une seule commande.**

```bash
git clone https://github.com/DLuffy07/inferloop.git && cd inferloop
docker compose up --build
```

Aucun fichier `.env` n'est requis (toutes les variables ont une valeur par défaut). Au **premier** démarrage, l'API
télécharge le modèle (~1 Go) dans le volume `hf-cache`. Les démarrages suivants prennent quelques secondes.

| Service | URL | Rôle |
|---|---|---|
| nginx | http://localhost | Point d'entrée public, rate limit 100 req/min/IP |
| api | http://localhost:8000/docs | FastAPI (Swagger), accès direct |
| grafana | http://localhost:3000 | Dashboard « InferLoop » + alertes (lecture seule sans login) |
| prometheus | http://localhost:9090 | Métriques, scraping toutes les 5 s |
| redis | interne (6379) | Cache des inférences, TTL 1 h |

## Utilisation

```bash
curl -s -X POST localhost/infer -H "Content-Type: application/json" \
  -d '{"titre": "Budget 2025 adopté", "texte": "L'\''Assemblée a adopté le PLF 2025 après le 49-3."}'
```
```json
{"categorie": "politique", "score": 0.81, "scores": {"politique": 0.81, "economie": 0.12, "...": "..."},
 "latence_ms": 142.3, "cache": "MISS", "modele": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"}
```

| Endpoint | Description | Codes |
|---|---|---|
| `POST /infer` | `{"titre"?, "texte"}` → catégorie, score, scores par classe, latence, HIT/MISS | 200 · 422 entrée invalide (vide, > 5 000 car.) · 503 modèle non chargé · 504 timeout |
| `GET /health` | Modèle chargé ? Redis joignable ? | 200 · 503 si modèle absent (utilisé par le healthcheck Docker) |
| `GET /metrics` | Format Prometheus | 200 |
| `POST /feedback` | Correction humaine `{"texte", "categorie_predite"?, "categorie_correcte"}` → JSONL pour re-training | 201 · 422 |

## Architecture

```
client ──► nginx :80 ──(round-robin, rate limit)──► api :8000 (×N) ──► mDeBERTa (volume hf-cache)
                                                        │   ▲
                                                        ▼   │ cache sha256(texte), TTL 1 h
                                                      redis :6379
prometheus :9090 ──scrape /metrics (découverte DNS de chaque réplica)──► api
grafana :3000 ──► prometheus  (datasource, dashboard et 4 alertes provisionnés au démarrage)
```

```
.
├── api/                  FastAPI : main.py (endpoints, métriques, cache), labels.py (catégories & prompts)
├── model/                Choix du modèle (README), baseline de drift, artefacts d'Adrien
├── monitoring/           prometheus/prometheus.yml · grafana/{provisioning,dashboards}
├── nginx/                nginx.conf (reverse proxy, rate limit, LB)
├── tests/                pytest, modèle et Redis mockés
├── loadtest/             locustfile.py + reports/
├── scripts/              evaluate_model.py (qualité/latence) · demo.py (scénarios de soutenance)
├── docs/                 RUNBOOK.md · matériaux J1
├── reports/              évaluations du modèle
├── Dockerfile            multi-stage builder → runtime, non-root
├── docker-compose.yml    5 services + locust (profil loadtest)
└── .github/workflows/ci.yml
```

## Choix du modèle

`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` en zero-shot. Le CamemBERT fine-tuné d'Adrien est livré **sans ses
poids**, et nous n'avons pas de corpus d'entraînement pour le reconstruire. mDeBERTa est le seul candidat multilingue
utilisable tout de suite.

Trois configurations mesurées via l'API sur les 500 articles de test (`scripts/eval_api.py`) :

| Configuration | Accuracy | F1 macro | p99 |
|---|---|---|---|
| **`simple` (retenue)** | **0,656** | **0,649** | 707 ms |
| `fr` (libellés descriptifs) | 0,614 | 0,582 | 762 ms |
| `simple` + int8 | 0,154 | 0,147 | 432 ms |

La quantification int8 est plus rapide mais détruit le modèle : rejetée. Le SLA p99 < 500 ms n'est pas tenu sur CPU
pour une requête isolée. Leviers et analyse : [`model/README.md`](model/README.md).

## Docker

- **Multi-stage** : le stage `builder` installe les dépendances dans un venv (pip, cache, headers torch restent dans ce stage), le stage `runtime` ne copie que le venv et le code. Torch **CPU uniquement**, modèle **hors image** (volume) → image nettement sous 2 Go (taille vérifiée en CI).
- Utilisateur non-root, `HEALTHCHECK` en Python (pas de curl dans l'image slim).
- Compose : healthcheck sur les 5 services, `depends_on: service_healthy` (redis → api → nginx, prometheus → grafana), `restart: unless-stopped`, volumes nommés pour Prometheus, Grafana, le cache modèle et le feedback, rotation des logs.
- Prometheus ne dépend pas d'une API saine : il doit tourner pour **observer** une API en panne.

```bash
docker images inferloop-api:local          # taille de l'image
docker compose ps                          # healthchecks
```

## Monitoring & alertes

Dashboard provisionné automatiquement (page d'accueil Grafana) : réplicas up, débit, latence p50/p95/p99 (HTTP et
modèle seul), erreurs 4xx/5xx, requêtes par réplica, cache hit rate, part de chaque classe vs baseline, score de
drift, liste des alertes, feedback.

| Alerte | Condition | Pendant |
|---|---|---|
| Latence p99 /infer > 500 ms | `histogram_quantile(0.99, …) > 0.5` | 30 s |
| Taux d'erreur /infer > 5 % | (4xx + 5xx) / total > 0,05 | 30 s |
| API down | `sum(up{job="inferloop-api"}) < 1` (0 si aucun réplica) | immédiat |
| Drift > 20 points *(bonus)* | écart max entre la part d'une classe (10 min) et la baseline > 0,20, dès 30 prédictions | immédiat |

Procédures de réponse : [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## CI/CD

`.github/workflows/ci.yml` : **lint** (`ruff check .`) → **tests** (`pytest` + couverture, rapports en artefact) ∥
**validation des configs** (`docker compose config`, `nginx -t`, `promtool`, JSON du dashboard) → **build** de l'image
taguée avec le SHA → **contrôle taille < 2 Go** → **smoke test** du conteneur → **push GHCR**
(`ghcr.io/dluffy07/inferloop:<sha>` et `:latest`) uniquement sur `main`. Le build est après les tests : on ne
construit ni ne pousse une image dont le code ne passe pas.

```bash
pip install -r requirements-dev.txt
ruff check . && pytest --cov=api
```

## Tests de charge

```bash
# Interface web Locust -> http://localhost:8089 (50 users, spawn 10/s)
docker compose --profile loadtest up locust

# Headless, rapport HTML + CSV dans loadtest/reports/
docker compose --profile loadtest run --rm locust -f /mnt/locust/locustfile.py --host http://nginx:8080 \
  --headless -u 50 -r 10 -t 3m --html /mnt/locust/reports/locust_report.html --csv /mnt/locust/reports/locust
```

Locust passe par un listener nginx **interne** (`:8080`, non publié sur l'hôte) : même load balancing, sans le rate
limit par IP, qui bloquerait sinon 50 utilisateurs venant de la même IP. Trafic : 70 % d'articles nouveaux (vraie
inférence), 30 % d'articles rejoués (cache hit). Les rejets 503 sont comptés à part (`/infer [délesté 503]`).

### Résultats (50 utilisateurs, 3 min, Docker Desktop 4 cœurs CPU)

La demande (~30 req/s) est ~15 fois au-dessus de la capacité d'inférence d'un CPU avec ce modèle. Le but n'est pas
de tout servir, mais de **se dégrader proprement**. Quatre itérations, chacune corrigeant ce que la précédente a révélé :

| Version | Nouveaux servis | Latence (p50 / p99) | 504 | Déjà vus OK | Problème révélé |
|---|---|---|---|---|---|
| 1. Threadpool FastAPI (40 threads) | 20 | ~10 s | 771 | 9 % | Sur-souscription CPU : 40 inférences à 4 threads sur 4 cœurs, le cache bloqué derrière |
| 2. Pool dédié 1 worker + file 8, timeout 5 s | 96 (0,5/s) | — | 208 | 70 % | Attente max ≈ timeout : ~75 % du CPU sur des réponses déjà abandonnées |
| 3. File 4, timeout 10 s, 1 x 4 threads | 319 (1,78/s) | 2,6 s / 3,1 s | **0** | **100 %** | — |
| **4. 2 inférences x 2 threads** (retenue) | **437 (2,43/s)** | **2,3 s / 3,0 s** | **0** | **100 %** | — |

Dans la version retenue, le cache est servi en 7 ms médian et les 3 272 requêtes excédentaires sont rejetées en 6 ms
avec `Retry-After: 2`. L'API reste `healthy` du début à la fin. Sur les nouveaux articles, la latence est surtout de
l'attente en file : une inférence seule prend ~0,5 s. Pour absorber 30 req/s, il faudrait ~12 fois plus de calcul
(GPU, ONNX Runtime, modèle distillé) : voir [`model/README.md`](model/README.md).

Rapports HTML : `loadtest/reports/locust_report.html` (version 3) et `loadtest/reports/locust_report_2x2.html` (version 4).

## Démo de soutenance

```bash
python scripts/demo.py smoke        # 10 articles réels -> catégorie, score, latence, HIT/MISS
python scripts/demo.py cache        # MISS puis HIT sur le même article
python scripts/demo.py ratelimit    # rafale sur :80 -> 429
python scripts/demo.py errors       # déclenche l'alerte taux d'erreur
python scripts/demo.py drift        # déclenche l'alerte drift
python scripts/demo.py feedback     # POST /feedback

# Résilience : suivre /health, puis simuler un crash du process
python scripts/demo.py resilience
docker compose exec api python -c "import os,signal; os.kill(1, signal.SIGTERM)"
# -> conteneur relancé par la restart policy, alerte "API down" visible dans Grafana, retour en < 30 s
```

> `docker compose stop api` est un arrêt volontaire : Docker **n'applique pas** la restart policy dans ce cas. Pour
> montrer le redémarrage automatique, il faut faire crasher le process (commande ci-dessus).

### Scaling horizontal (bonus)

```bash
# 3 réplicas sur 4 cœurs : 1 inférence x 1 thread chacun (sinon ils se disputent le CPU)
printf "MAX_CONCURRENT_INFERENCES=1\nTORCH_THREADS=1\n" > .env
docker compose -f docker-compose.yml up -d --scale api=3
# Toujours passer -f docker-compose.yml (sinon Compose revient à 1 réplica) et --no-deps
docker compose -f docker-compose.yml --profile loadtest run --rm --no-deps locust \
  -f /mnt/locust/locustfile.py --host http://nginx:8080 --headless -u 50 -r 10 -t 2m --only-summary
```
Sans `docker-compose.override.yml`, l'API n'a plus de port publié et peut donc être répliquée. nginx
(`server api:8000 resolve`) fait du round-robin sur les 3 réplicas, et Prometheus les découvre par DNS. La
répartition apparaît dans le panneau « Requêtes/s par réplica ». Chaque réplica charge son propre modèle en mémoire
(~1,5 Go chacun) : prévoir au moins 6 Go de RAM pour Docker.

Sur une seule machine, les réplicas se partagent les mêmes cœurs : le débit total ne dépasse pas celui d'un réplica
bien réglé. Le gain est la **disponibilité** : si un réplica crashe, les deux autres continuent de servir, là où un
réplica seul coupe le service pendant ~8 s.

## Configuration

Toutes les variables sont documentées dans [`.env.example`](.env.example) (`cp .env.example .env`). Aucun secret n'est
commité : `.env` est ignoré par git, et le mot de passe admin Grafana par défaut est celui de Grafana (`admin`), à
surcharger via `GRAFANA_ADMIN_PASSWORD`.

## Limites connues & pistes

- Qualité zero-shot (65,6 % d'accuracy mesurée) inférieure au CamemBERT fine-tuné d'Adrien (87 % annoncé). La vraie
  suite serait de fine-tuner un `distilcamembert` sur les retours `/feedback` et un corpus annoté.
- p99 à 707 ms sur CPU, au-dessus du SLA de 500 ms. La quantification int8 a été testée et rejetée (accuracy 0,15). Leviers : cache, CPU, `--scale`, puis export ONNX Runtime ou GPU.
- Mise à jour du modèle sans interruption : il faudrait un déploiement blue/green (deux services api derrière nginx),
  ou passer sur Kubernetes (rolling update, HPA).
