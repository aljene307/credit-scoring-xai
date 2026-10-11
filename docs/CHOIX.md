# Journal des choix

Pour chaque décision : le choix retenu, la raison, et l'alternative écartée.
Statut : **retenu**, **à confirmer** (dépend d'une vérification sur les
données) ou **en attente** (sera tranché à une étape ultérieure).

---

## Cadrage

### C01. Cas métier : scoring de crédit (cas n°1 du catalogue) — retenu
- **Choix :** prédire le défaut de paiement d'un client de carte de crédit.
- **Raison :** cas d'usage central en banque, où l'explicabilité est une
  exigence réglementaire (justifier un refus au client et au régulateur).
- **Alternative écartée :** les autres cas du catalogue.

### C02. Jeu de données : Default of Credit Card Clients (Taiwan, UCI 350) — retenu
- **Choix :** téléchargement automatique avec `ucimlrepo.fetch_ucirepo(id=350)`.
- **Raison :** 30 000 clients, 23 variables, pas de valeurs manquantes annoncées,
  licence CC BY 4.0, téléchargement sans intervention manuelle.
- **Alternatives écartées :** German Credit (1 000 lignes seulement) ;
  Give Me Some Credit (Kaggle, nécessite un compte et une clé d'API).

### C03. Renommage des colonnes X1…X23 en noms lisibles — retenu
- **Choix :** LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2…PAY_6,
  BILL_AMT1…6, PAY_AMT1…6.
- **Raison :** graphiques et explications lisibles par un décideur.
- **Alternative écartée :** garder X1…X23 (explications incompréhensibles).

### C04. Modalités non documentées de EDUCATION, MARRIAGE et PAY_* — à confirmer
- **Choix :** vérifier d'abord leur présence et leur fréquence sur les vraies
  données, puis décider du recodage.
- **Options :** regrouper dans « autres », conserver telles quelles, supprimer
  les lignes concernées.

---

## Préparation des données

### P01. Variables catégorielles déclarées explicitement — retenu
- **Choix :** pour le jeu Taiwan, liste explicite des variables catégorielles
  (SEX, EDUCATION, MARRIAGE), encodées en one-hot. Pour un CSV importé,
  l'utilisateur choisit les variables catégorielles dans l'interface.
- **Raison :** dans ce jeu, toutes les colonnes sont des entiers, y compris
  SEX, EDUCATION et MARRIAGE. Une détection par type les traiterait comme
  des nombres et supposerait un ordre qui n'existe pas (marié < célibataire).
- **Alternative écartée :** détection automatique par type de colonne.

### P02. PAY_0 … PAY_6 traitées comme numériques ordinales — retenu
- **Choix :** les variables de retard de paiement restent numériques.
- **Raison :** leurs valeurs ont un ordre naturel (plus de mois de retard =
  plus de risque) ; le one-hot multiplierait les colonnes et perdrait cet ordre.
- **Alternative écartée :** one-hot de chaque modalité de PAY_*.

### P03. Prétraitement dans un Pipeline à l'intérieur de GridSearchCV — retenu
- **Choix :** `Pipeline(ColumnTransformer → modèle)` passé à `GridSearchCV`.
  Le ColumnTransformer applique l'imputation, le one-hot (catégorielles) et la
  standardisation (numériques).
- **Raison :** l'encodeur et la standardisation sont réappris sur chaque pli
  d'entraînement ; aucune information des plis de validation ne fuit dans le
  réglage (consigne de l'enseignante).
- **Alternative écartée :** prétraiter tout le jeu avant la validation croisée
  (fuite de données).

### P04. Séparation entraînement / test stratifiée — retenu
- **Choix :** 80 % / 20 % par défaut (paramétrable), stratifiée sur la cible,
  graine fixe.
- **Raison :** même proportion de défauts dans les deux jeux, résultats
  reproductibles. Le jeu de test ne sert jamais au réglage.
- **Alternative écartée :** séparation aléatoire simple.

---

## Modélisation

### M01. Quatre modèles — retenu
- **Choix :** régression logistique et arbre de décision (basiques),
  Random Forest et XGBoost (ensemblistes).
- **Raison :** consigne de l'enseignante (elle prime sur le sujet) ; couvre le
  compromis interprétabilité / performance.
- **Alternative écartée :** k plus proches voisins (retiré par l'enseignante).

### M02. Optimisation par GridSearchCV, validation croisée stratifiée à 5 plis — retenu
- **Choix :** `GridSearchCV` sur le jeu d'entraînement seulement, score AUC-ROC.
- **Raison :** l'AUC mesure la capacité à classer les clients du plus au moins
  risqué, indépendamment du seuil, ce qui est l'usage d'un score de crédit.
- **Alternatives écartées :** F1 (dépend du seuil) ; PR-AUC (affichée mais pas
  utilisée pour le réglage) ; recherche aléatoire (consigne : GridSearchCV).

### M03. Pondération des classes testée dans la grille — retenu
- **Choix :** `class_weight ∈ {None, "balanced"}` (scikit-learn) et
  `scale_pos_weight ∈ {1, n_négatifs / n_positifs}` (XGBoost) font partie de
  la grille ; GridSearchCV décide.
- **Raison :** environ 22 % de défauts ; on laisse la validation croisée
  trancher plutôt que d'imposer la pondération.
- **Effet à connaître :** la pondération gonfle les probabilités prédites
  (moyenne proche de 50 % au lieu de 22 %). Elle change peu le classement des
  clients, donc peu l'AUC, mais les probabilités ne sont plus des
  probabilités de défaut réelles.
- **Alternative écartée :** SMOTE (crée des clients synthétiques, dépendance
  supplémentaire, explications SHAP sur des données artificielles).

### M04. Modèles de référence entraînés hors ligne — retenu
- **Choix :** un script entraîne et sauvegarde les modèles ; l'interface permet
  aussi de relancer un GridSearch avec des grilles réduites.
- **Raison :** un GridSearch complet sur 30 000 lignes prend plusieurs minutes
  et dépasse les ressources de Streamlit Community Cloud.
- **Alternative écartée :** tout entraîner à chaque ouverture de l'application.

---

## Évaluation

### E01. Seuil de décision — retenu
- **Choix :** le seuil recommandé est choisi sur les prédictions de validation
  croisée du jeu d'entraînement (`cross_val_predict`), puis évalué une seule
  fois sur le test. Le curseur de seuil de l'interface reste sur le jeu de
  test, pour la démonstration uniquement.
- **Raison :** choisir le seuil sur le test reviendrait à régler le modèle sur
  le test et rendrait l'évaluation finale trop optimiste.
- **Alternative écartée :** choisir le seuil en regardant le test.
- **En attente :** critère de choix du seuil (F1 maximal, point du KS, coût
  métier).

### E02. Métriques — retenu
- **Choix :** AUC-ROC, précision, rappel et F1 sur la classe « défaut »,
  PR-AUC, indice de Gini (2 × AUC − 1) et statistique de Kolmogorov-Smirnov.
- **Raison :** demandées par le sujet pour le scoring de crédit ; Gini et KS
  sont les indicateurs usuels en banque.

---

## Explicabilité

### X01. Valeurs SHAP regroupées par variable d'origine — retenu
- **Choix :** les valeurs SHAP des colonnes issues du one-hot (par exemple
  EDUCATION_1, EDUCATION_2…) sont additionnées par variable d'origine avant
  affichage.
- **Raison :** le décideur raisonne sur « le niveau d'études », pas sur une
  colonne technique ; la somme est valide car les valeurs SHAP sont additives.
- **Alternative écartée :** afficher chaque colonne one-hot séparément.

---

## Architecture et outillage

### A01. Code partagé dans un paquet `src/credit_scoring` — retenu
- **Choix :** le notebook, l'application et l'API importent les mêmes fonctions.
- **Raison :** un seul calcul pour chaque métrique ou explication ; testable.
- **Alternative écartée :** un seul gros fichier Streamlit, recopié dans l'API.

### A02. Une page Streamlit par module — retenu
- **Raison :** suit l'ordre du sujet, facilite la présentation orale.

### A03. Python 3.11 — retenu
- **Raison :** version éprouvée pour SHAP, XGBoost et LIME, disponible sur
  Streamlit Community Cloud et dans les images Docker.
- **Alternative écartée :** Python 3.13 (plus de risques d'incompatibilité).

### A04. Fins de ligne LF dans le dépôt (`.gitattributes`) — retenu
- **Raison :** le déploiement se fait sous Linux ; évite les différences
  parasites entre Windows et Linux.

### A05. Modèles entraînés hors du dépôt pour l'instant — en attente
- **Choix :** `models/` est ignoré par Git tant que la taille des fichiers
  n'est pas connue.
- **À trancher au déploiement :** commiter les modèles, les réentraîner à la
  construction de l'image Docker, ou les héberger ailleurs.

### A06. Dépôt GitHub privé jusqu'à la vérification finale — retenu
- **Raison :** l'historique est contrôlé (secrets, données personnelles) avant
  publication.

### A07. Licence MIT pour le code — retenu
- **Raison :** licence simple et permissive. Le jeu de données reste sous sa
  propre licence CC BY 4.0, citée dans le README.
