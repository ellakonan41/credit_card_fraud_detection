"""Cache Redis pour l'idempotence des prédictions.

Si une même transaction est soumise plusieurs fois (ex. retry réseau côté
client), on renvoie le résultat déjà calculé plutôt que de relancer
l'inférence et de risquer un double traitement en aval.

Le cache est **optionnel** :
- activé si la variable d'environnement `REDIS_URL` est définie (cas
  docker-compose) ;
- sinon complètement inactif (aucun appel réseau) — l'API fonctionne
  normalement, chaque transaction est simplement re-scorée.
"""

import hashlib
import json
import os

import redis

_REDIS_URL = os.getenv("REDIS_URL")
_TTL_SECONDS = 300  # durée de vie d'une entrée : 5 minutes

# redis-py ne se connecte pas ici : la connexion est établie au 1er appel.
# Chaque opération est protégée par try/except, donc une panne de Redis ne
# bloque jamais l'API.
_client = (
    redis.from_url(_REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
    if _REDIS_URL
    else None
)


def _key(payload: dict) -> str:
    """Empreinte unique et stable d'une transaction (mêmes valeurs -> même clé)."""
    raw = json.dumps(payload, sort_keys=True).encode()
    return "pred:" + hashlib.sha256(raw).hexdigest()


def get_cached(payload: dict) -> dict | None:
    """Retourne le résultat en cache pour cette transaction, ou None."""
    if _client is None:
        return None
    try:
        hit = _client.get(_key(payload))
        return json.loads(hit) if hit else None
    except redis.RedisError:
        return None


def set_cached(payload: dict, result: dict) -> None:
    """Stocke le résultat pour cette transaction (expire après _TTL_SECONDS)."""
    if _client is None:
        return
    try:
        _client.setex(_key(payload), _TTL_SECONDS, json.dumps(result))
    except redis.RedisError:
        pass
