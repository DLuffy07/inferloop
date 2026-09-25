"""Catégories MédiaLens et configurations de prompt zero-shot.

Partagé entre l'API (api/main.py) et le script d'évaluation
(scripts/evaluate_model.py) pour garantir qu'on évalue exactement
ce qui tourne en production.
"""

# Identifiants canoniques = labels du dataset articles_test.csv
CATEGORIES = ["politique", "economie", "sport", "culture", "faits_divers"]

# Chaque mode associe un identifiant canonique -> texte candidat envoyé au modèle NLI.
PROMPT_MODES = {
    # Configuration mesurée J1 (accuracy 0.656 / F1 macro 0.649 sur 500 articles)
    "simple": {
        "template": "This example is {}.",
        "labels": {
            "politique": "politique",
            "economie": "économie",
            "sport": "sport",
            "culture": "culture",
            "faits_divers": "faits divers",
        },
    },
    # Template français + libellés descriptifs : lève les ambiguïtés
    # politique/économie et faits divers/justice.
    "fr": {
        "template": "Cet article parle de {}.",
        "labels": {
            "politique": "politique, gouvernement et élections",
            "economie": "économie, finance et entreprises",
            "sport": "sport et compétitions sportives",
            "culture": "culture, arts, cinéma et musique",
            "faits_divers": "faits divers, justice, accidents et criminalité",
        },
    },
}


def get_prompt(mode: str) -> tuple[str, dict[str, str]]:
    """Retourne (template, {texte_candidat: id_canonique}) pour un mode."""
    if mode not in PROMPT_MODES:
        raise ValueError(f"PROMPT_MODE inconnu : {mode} (attendu : {list(PROMPT_MODES)})")
    cfg = PROMPT_MODES[mode]
    reverse = {text: cat for cat, text in cfg["labels"].items()}
    return cfg["template"], reverse


def build_input(texte: str | None, titre: str | None = None) -> str:
    """Titre + texte, comme dans l'évaluation (Adrien : +3 pts F1 avec le titre)."""
    titre = (titre or "").strip()
    texte = (texte or "").strip()
    if titre and texte:
        return f"{titre}. {texte}"
    return titre or texte
