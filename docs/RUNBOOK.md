# Runbook d'exploitation — InferLoop

Accès rapides : API `http://localhost` (nginx) ou `http://localhost:8000` (direct) · Grafana `http://localhost:3000`
(lecture seule sans login, admin via `.env`) · Prometheus `http://localhost:9090` · Alertes : Grafana → Alerting.

## Diagnostic en 1 minute

```bash
docker compose ps                                    # état + healthchecks des 5 services
curl -s localhost/health                             # model_loaded, redis_connected
docker compose logs --tail=100 api                   # erreurs de chargement / d'inférence
curl -s localhost:9090/api/v1/targets | grep health  # scraping Prometheus
```

| Symptôme | Cause probable | Action |
|---|---|---|
| `api` bloqué en `health: starting` au 1er lancement | Téléchargement du modèle (~1 Go) dans le volume `hf-cache` | Attendre (start_period 300 s), suivre `docker compose logs -f api` |
| `/health` → 503, `model_loaded:false` | Hub HuggingFace injoignable, disque plein, mémoire insuffisante | Logs `api`, `docker system df`, mémoire Docker Desktop ≥ 4 Go |
| `redis_connected:false` | Redis arrêté | `docker compose up -d redis`. L'API continue sans cache (mode dégradé) |
| 429 côté client | Rate limit nginx (100 req/min/IP + rafale de 20) | Normal. Pour un test de charge, passer par le listener interne `nginx:8080` (profil `loadtest`) |
| 502/504 via nginx | Aucun réplica sain / inférence > 10 s | `docker compose ps api`, voir « Latence » |

## Alertes

### Latence p99 > 500 ms
1. Dashboard → « Latence /infer » vs « Latence modèle seul » : si les deux montent, le CPU est saturé.
2. `docker stats` : CPU de `api` à 100 % ?
3. Remèdes, du plus rapide au plus lourd : `QUANTIZE=1` dans `.env` puis `docker compose up -d api` ·
   `docker compose -f docker-compose.yml up -d --scale api=3` · limiter `TORCH_THREADS` (1 à 2 par réplica) quand il y a
   plusieurs réplicas, pour éviter qu'ils se disputent les cœurs.

### Taux d'erreur > 5 %
1. Panneau « Requêtes/s par code HTTP » : 4xx (payloads invalides côté client) ou 5xx (API) ?
2. 422 : contacter le client (schéma : `{"titre": "...", "texte": "..."}`, texte ≤ 5 000 caractères).
3. 503 : modèle non chargé, voir `/health`. 504 : timeout d'inférence (`INFERENCE_TIMEOUT`), voir « Latence ».
4. 500 : `docker compose logs api | grep -i error`.

### API down
1. `docker compose ps api` : `restarting` ? La restart policy `unless-stopped` relance après un crash.
2. Si l'API boucle : `docker compose logs --tail=200 api` (OOM → augmenter la mémoire Docker).
3. Retour attendu en moins de 30 s (modèle déjà en cache dans le volume).
4. ⚠️ `docker compose stop api` est un arrêt **volontaire** : Docker n'applique pas la restart policy. Relancer avec
   `docker compose start api`.

### Drift des prédictions > 20 points
1. Dashboard → « Actuel vs baseline » : quelle classe dévie ?
2. Vrai changement d'actualité (élections, JO…) → drift de données attendu, noter l'événement.
3. Glissement sans raison éditoriale → échantillonner les articles récents, les faire corriger via `POST /feedback`,
   mesurer l'accuracy sur ces retours (`data/feedback.jsonl` dans le volume `feedback-data`).
4. Après changement de modèle ou de prompt : régénérer la baseline (`scripts/evaluate_model.py --write-baseline`).

## Opérations courantes

```bash
# Mise à jour du code / modèle (le volume hf-cache est conservé)
git pull && docker compose up -d --build api

# Vider le cache Redis (après changement de modèle, inutile : la clé inclut modèle + prompt)
docker compose exec redis redis-cli FLUSHDB

# Exporter le feedback humain
docker compose cp api:/data/feedback.jsonl ./feedback.jsonl

# Repartir de zéro (supprime modèle téléchargé, métriques, dashboards modifiés)
docker compose down -v
```
