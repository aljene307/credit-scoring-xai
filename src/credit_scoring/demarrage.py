"""Premier import des bibliothèques lourdes, une seule fois, par un seul fil d'exécution.

Streamlit exécute chaque session dans son propre fil. Au premier démarrage, deux exécutions
simultanées (le contrôle de santé de l'hébergeur et un visiteur, par exemple) pouvaient importer
scikit-learn et scipy en même temps : l'une voyait alors scipy à moitié chargé et l'application
plantait. Python exécute le corps d'un module une seule fois et fait attendre les autres fils
jusqu'à la fin : importer ce module au début de app/Accueil.py suffit donc à ce que tout soit
chargé en une fois, avant toute page.
"""

import scipy  # noqa: F401
import scipy.stats  # noqa: F401
import sklearn  # noqa: F401

import credit_scoring.explain  # noqa: F401  (shap, lime)
import credit_scoring.fairness  # noqa: F401
import credit_scoring.global_explanations  # noqa: F401
import credit_scoring.models  # noqa: F401  (xgboost)
import credit_scoring.plots  # noqa: F401
import credit_scoring.training  # noqa: F401
