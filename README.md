# Scoring de crédit explicable

Application web de classification explicable appliquée au scoring de crédit :
prédire le défaut de paiement d'un client de carte de crédit, comparer
plusieurs modèles et expliquer leurs décisions (SHAP, LIME).

Projet de Data Mining, ENSI, 2026-2027.

> Projet en cours de développement.

## Jeu de données

**Default of Credit Card Clients** (Taiwan, 2005), UCI Machine Learning
Repository, identifiant 350. 30 000 clients, 23 variables explicatives,
cible : défaut de paiement le mois suivant.

- Source : <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- Licence : [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Référence : Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining
  techniques for the predictive accuracy of probability of default of credit
  card clients. *Expert Systems with Applications*, 36(2), 2473-2480.

Le jeu de données est téléchargé automatiquement par l'application ; il n'est
pas inclus dans ce dépôt.

## Décisions de conception

Voir [docs/CHOIX.md](docs/CHOIX.md).

## Licence

Code sous licence MIT, voir [LICENSE](LICENSE).
