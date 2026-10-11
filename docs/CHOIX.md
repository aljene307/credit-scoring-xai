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

### C04. Modalités non documentées : vérification puis recodage — retenu
- **Choix :** vérifier d'abord leur présence et leur fréquence sur les vraies
  données (fait, voir le notebook, section 1), puis décider (R1 à R5).
- **Mise en œuvre :** le recodage applique une règle fixe ligne par ligne et
  n'apprend rien des données ; il est donc appliqué après chargement, avant
  la séparation entraînement / test, sans risque de fuite. Le cache conserve
  les données brutes.

**Codes officiels (UCI) :** SEX 1 = homme, 2 = femme ; EDUCATION 1 = 3e cycle,
2 = université, 3 = lycée, 4 = autres ; MARRIAGE 1 = marié, 2 = célibataire,
3 = autres ; PAY -1 = payé à temps, 1…9 = mois de retard.

#### R1. EDUCATION : codes 0, 5 et 6 regroupés avec 4, libellé « Autres ou inconnu » — retenu
- **Constat :** 0 (14 clients, 0 % de défaut), 5 (280 ; 6,4 %), 6 (51 ; 15,7 %),
  soit 345 clients (1,2 %) ; le code 4 « autres » compte 123 clients (5,7 %).
- **Sources :** l'UCI ne documente pas 0, 5 et 6. La description du même jeu
  sur Kaggle (« Default of Credit Card Clients Dataset », UCI Machine
  Learning, <https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset>)
  indique 5 = unknown et 6 = unknown ; le code 0 n'est documenté nulle part.
- **Raison :** codes sans sens officiel, effectifs trop faibles pour être
  appris (le code 0 peut manquer de certains plis de validation croisée),
  risque proche de celui de « autres ». Le décideur retrouve les 4 modalités
  officielles.
- **Alternatives écartées :** modalité « inconnu » séparée ; garder les
  7 codes ; supprimer les 345 lignes.

#### R2. MARRIAGE : code 0 regroupé avec 3 (« autres ») — retenu
- **Constat :** code 0 = 54 clients, 5 défauts (9,3 %) ; code 3 = 323 clients,
  84 défauts (26,0 %).
- **Test statistique :** test exact de Fisher bilatéral, code 0 contre code 3 :
  **p = 0,0055**. L'écart est statistiquement significatif au seuil de 5 %.
  Intervalles de confiance à 95 % (Wilson) : code 0 [4,0 % ; 19,9 %],
  code 3 [21,5 % ; 31,1 %].
- **Raison :** le code 0 n'a pas de sens documenté, et 54 clients sur 30 000
  (0,18 %) ne peuvent pas peser sur le modèle. Le regroupement n'est pas
  justifié par l'absence d'écart : l'écart existe.
- **Conséquence :** le taux de défaut de « autres » passe de 26,0 % à 23,6 %
  (89 / 377).
- **Alternatives écartées :** modalité « inconnu » séparée ; supprimer les
  54 lignes.

#### R3. PAY_* : aucun recodage — retenu
- **Constat :** -2 et 0 sont fréquents (PAY_0 : 2 759 et 14 737 clients).
  Interprétation la plus répandue : -2 = pas d'utilisation, -1 = payé en
  totalité, 0 = crédit renouvelable (minimum payé). Indice dans les données :
  32,7 % des PAY_0 = -2 ont une facture nulle ou négative, contre 0 % des
  PAY_0 = 0. Taux de défaut selon PAY_0 : -2 : 13,2 % ; -1 : 16,8 % ;
  0 : 12,8 % ; 1 : 33,9 % ; 2 : 69,1 %.
- **Raison :** valeurs conservées de -2 à 8 ; l'application affiche un libellé
  pour chaque code. Ramener -2 et -1 à 0 perdrait de l'information et
  imposerait une interprétation non officielle.
- **Limite à citer :** la valeur 1 concerne 3 688 clients en septembre (PAY_0)
  mais presque aucun les autres mois (28, 4, 2, 0, 0) : le codage a
  probablement changé d'un mois à l'autre.
- **Alternatives écartées :** ramener -2, -1 et 0 à 0 ; séparer en deux
  variables (statut, mois de retard).

#### R4. Clients aux variables identiques conservés — retenu
- **Constat :** 52 groupes (108 lignes) ont les mêmes 23 variables avec des
  identifiants différents, dont 21 groupes avec des cibles différentes ;
  ce sont surtout des comptes inactifs.
- **Raison :** clients distincts au profil identique (0,36 % des lignes).
- **Alternative écartée :** supprimer les doublons exacts.

#### R5. Montants négatifs et dépassements de plafond conservés — retenu
- **Constat :** environ 600 à 690 factures négatives par mois (soldes
  créditeurs, minimum -165 580) ; 2 115 clients ont une facture supérieure à
  leur plafond.
- **Raison :** situations bancaires réelles (trop-perçu, intérêts et frais).
- **Alternative écartée :** plafonner ou supprimer ces valeurs.

**Observation pour l'analyse critique :** 795 clients ont toutes leurs
factures et tous leurs paiements à 0, et 37,6 % d'entre eux font défaut
(contre 22,1 % en moyenne) : définition du défaut plus large qu'un impayé,
ou bruit dans les étiquettes.

### C05. Cible renommée DEFAULT — retenu
- **Choix :** la colonne `Y` devient `DEFAULT` (1 = défaut de paiement le mois
  suivant, 0 = pas de défaut).
- **Raison :** nom explicite dans les graphiques et les explications.
- **Alternative écartée :** garder `Y`, ou le nom long d'origine
  (« default payment next month »).

### C06. Cache local des données au format Parquet — retenu
- **Choix :** au premier lancement, le jeu est téléchargé puis enregistré dans
  `data/` (ignoré par Git) ; les lancements suivants lisent ce fichier.
- **Raison :** le serveur de l'UCI est parfois lent ou indisponible. Parquet
  conserve les types de colonnes et se lit vite ; pyarrow est déjà requis par
  Streamlit.
- **Alternative écartée :** CSV (lisible à l'œil mais perd les types) ;
  télécharger à chaque lancement.

### C07. Notebook commité avec ses sorties — retenu
- **Choix :** le notebook est exécuté en entier puis commité avec ses
  graphiques et ses chiffres, après contrôle des chemins locaux dans les
  sorties.
- **Raison :** les résultats sont lisibles directement sur GitHub, sans
  réexécution.
- **Alternative écartée :** notebook sans sorties (plus léger, mais il faut
  le réexécuter pour voir les résultats).

### C10. Graphiques : Matplotlib dans le notebook, Plotly dans l'application — retenu
- **Choix :** les graphiques du notebook sont faits avec Matplotlib (images
  fixes) ; ceux de l'application avec Plotly (interactifs : survol, zoom).
- **Raison :** GitHub n'affiche pas les graphiques Plotly d'un notebook (ils
  apparaissent vides) ; Matplotlib sera de toute façon installé avec SHAP.
- **Alternative écartée :** Plotly avec export en images par kaleido
  (dépendance supplémentaire, parfois difficile à installer sous Windows).
- **Installation :** Matplotlib est dans `requirements-dev.txt` tant que seul
  le notebook l'utilise ; il passera dans `requirements.txt` avec SHAP.
- **Lisibilité :** les tranches de quantiles sont libellées par les valeurs
  réellement observées (« 10 000 – 30 000 ») ; dans le notebook, les barres
  reposant sur moins de 30 clients sont grisées (taux instable).

### C11. Couleurs des graphiques — retenu
- **Choix :** bleu `#2a78d6` pour « pas de défaut », orange `#eb6834` pour
  « défaut » ; corrélations sur une échelle bleu ↔ gris ↔ rouge centrée sur 0.
- **Raison :** paire de couleurs distinguable par les personnes daltoniennes ;
  une échelle divergente avec un milieu neutre montre le signe de la
  corrélation sans arc-en-ciel.
- **Alternative écartée :** les couleurs par défaut de Plotly (vert et rouge
  sont confondus par les daltoniens les plus fréquents).

### C08. Renommage contrôlé par les métadonnées UCI — retenu
- **Choix :** la correspondance X1…X23 → noms lisibles est écrite dans le code,
  et à chaque téléchargement elle est comparée aux descriptions fournies par
  l'UCI (`fetch_ucirepo(id=350).variables`). En cas d'écart, le chargement
  s'arrête avec un message explicite.
- **Raison :** les métadonnées UCI donnent bien X1 = LIMIT_BAL … X6 = PAY_0,
  X7 = PAY_2 … X23 = PAY_AMT6 (il n'existe pas de PAY_1 dans le jeu d'origine).
  Si l'UCI changeait l'ordre des colonnes, un renommage non contrôlé
  inverserait des variables sans aucune erreur visible.
- **Alternative écartée :** renommer directement à partir des descriptions
  UCI (le code dépendrait du réseau, y compris dans les tests).

### C09. Identifiant client (ID) conservé comme index — retenu
- **Choix :** la colonne `ID` fournie par l'UCI devient l'index du tableau ;
  elle n'est jamais utilisée comme variable explicative.
- **Raison :** permet de désigner un client précis dans l'application
  (explication locale) ; un numéro d'ordre n'a aucun sens prédictif.
- **Alternative écartée :** supprimer l'identifiant.

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
  Le ColumnTransformer applique l'imputation (médiane pour les numériques,
  modalité la plus fréquente pour les catégorielles), le one-hot
  (catégorielles) et, pour la régression logistique seulement, la
  standardisation (P05).
- **Raison :** l'imputation, l'encodeur et la standardisation sont réappris
  sur chaque pli d'entraînement ; aucune information des plis de validation ne
  fuit dans le réglage (consigne de l'enseignante).
- **Alternative écartée :** prétraiter tout le jeu avant la validation croisée
  (fuite de données).

### P04. Séparation entraînement / test stratifiée — retenu
- **Choix :** 80 % / 20 % par défaut (paramétrable de 10 à 40 %), stratifiée
  sur la cible, graine 42 (modifiable).
- **Raison :** même proportion de défauts dans les deux jeux, résultats
  reproductibles. La séparation est faite avant tout apprentissage ; le jeu de
  test ne sert jamais au réglage.
- **Alternatives écartées :** séparation aléatoire simple ; trois jeux
  entraînement / validation / test (la validation croisée joue déjà le rôle de
  jeu de validation) ; séparation chronologique (pas de date par client).

### P05. Standardisation pour la régression logistique seulement — retenu
- **Choix :** les variables numériques sont standardisées (moyenne 0,
  écart-type 1, `StandardScaler`) uniquement dans le Pipeline de la régression
  logistique. L'arbre de décision, Random Forest et XGBoost reçoivent les
  variables numériques dans leur unité d'origine (imputation seulement).
- **Raison :** la pénalité de la régression logistique traite tous les
  coefficients de la même façon, ce qui suppose des variables à la même
  échelle. Les arbres ne dépendent pas de l'échelle : la standardisation ne
  changerait pas leurs prédictions, mais rendrait illisibles pour un décideur
  les règles de l'arbre (« LIMIT_BAL ≤ -0,42 » au lieu de « LIMIT_BAL ≤ 75 000 »)
  et les graphiques SHAP.
- **Alternatives écartées :** un prétraitement identique pour les quatre
  modèles (plus simple, mais explications en unités standardisées) ;
  MinMaxScaler (très sensible aux montants extrêmes) ; transformation par
  quantiles dans la grille de la régression logistique (GridSearch plus long).

### P06. One-hot sans suppression de modalité, modalités inconnues ignorées — retenu
- **Choix :** `OneHotEncoder(drop=None, handle_unknown="ignore")`.
- **Raison :** avec la pénalité de la régression logistique, garder toutes les
  modalités ne pose pas de problème de colinéarité, et le regroupement des
  valeurs SHAP par variable d'origine (X01) est direct. Une modalité absente de
  l'entraînement (CSV importé, saisie via l'API) donne des zéros au lieu d'une
  erreur.
- **Alternatives écartées :** supprimer la première modalité (`drop="first"`) ;
  target encoding (inutile avec 2 à 4 modalités) ; encodage ordinal (impose un
  ordre).

### P07. Pas de nouvelles variables pour l'instant — retenu
- **Choix :** les modèles utilisent les 23 variables d'origine.
- **Raison :** explications directement compréhensibles ; on mesure d'abord.
  Si la régression logistique souffre de la non-linéarité de PAY_0 (M05),
  l'indicateur « retard ≥ 1 mois » sera la première amélioration testée.
- **Alternative écartée :** ajouter dès maintenant un taux d'utilisation
  (facture / plafond), un nombre de mois en retard, un indicateur de retard
  (valables pour le jeu Taiwan seulement).

### P08. Variables sensibles gardées par défaut, exclusion possible — retenu
- **Choix :** SEX, AGE, MARRIAGE et EDUCATION sont utilisées par défaut ; la
  page Préparation permet de les exclure.
- **Raison :** l'analyse d'équité (module 5) compare les modèles avec et sans
  ces variables ; les exclure d'office empêcherait de mesurer leur effet.
- **Alternative écartée :** les exclure d'office.

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
  Les 5 plis sont stratifiés, mélangés avec la graine 42, et identiques pour
  les quatre modèles. PR-AUC, F1, rappel et précision sont aussi calculés à
  chaque essai (tableau comparatif), et l'AUC sur les plis d'entraînement est
  conservée pour détecter le surapprentissage.
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
- **Résultat (validation croisée) :** la pondération est retenue pour la
  régression logistique, Random Forest et XGBoost, mais son effet sur l'AUC
  est négligeable : +0,0004, +0,0004 et +0,0001. Pour l'arbre, la meilleure
  combinaison sans pondération est meilleure de 0,0066. Comme prévu, la
  pondération change surtout les probabilités et donc le seuil utile.

### M04. Modèles de référence entraînés hors ligne — retenu
- **Choix :** un script entraîne et sauvegarde les modèles ; l'interface permet
  aussi de relancer un GridSearch avec des grilles réduites.
- **Raison :** un GridSearch complet sur 30 000 lignes prend plusieurs minutes
  et dépasse les ressources de Streamlit Community Cloud.
- **Alternative écartée :** tout entraîner à chaque ouverture de l'application.

### M05. Relation non linéaire entre PAY_0 et le risque — confirmé en partie (étape 3)
- **Résultat :** AUC de validation 0,728 pour la régression logistique contre
  0,767 à 0,785 pour les trois modèles à base d'arbres. Le meilleur `C` est la
  plus grande valeur de la grille (régularisation la plus faible) et l'AUC
  varie de moins de 0,002 entre `C` = 0,1 et 10 : le modèle n'est pas freiné
  par la régularisation mais limité par sa forme linéaire. Confirmation à
  chercher dans les graphiques de dépendance SHAP (étape 5).
- **Constat :** le taux de défaut est plat de PAY_0 = -2 à 0 (13 à 17 %), puis
  saute à 1 (33,9 %) et à 2 (69,1 %).
- **Conséquence attendue :** la régression logistique suppose un effet
  linéaire de PAY_0 sur le logit du risque ; elle peut donc être désavantagée
  par rapport aux arbres, qui découpent librement les seuils.
- **À vérifier à l'étape 3 :** comparer les performances, et regarder la forme
  de l'effet de PAY_0 dans les graphiques de dépendance SHAP.

### M06. Grilles d'hyperparamètres de référence — retenu
| Modèle | Grille | Combinaisons |
|---|---|---|
| Régression logistique | `C` ∈ {0,01 ; 0,1 ; 1 ; 10}, `class_weight` ∈ {aucun, balanced} | 8 |
| Arbre de décision | `max_depth` ∈ {3, 5, 8, 12}, `min_samples_leaf` ∈ {1, 20, 100}, `class_weight` ∈ {aucun, balanced} | 24 |
| Random Forest (300 arbres) | `max_depth` ∈ {8, 12, sans limite}, `min_samples_leaf` ∈ {1, 10}, `max_features` ∈ {sqrt, 0,5}, `class_weight` ∈ {aucun, balanced_subsample} | 24 |
| XGBoost | `n_estimators` ∈ {200, 400}, `learning_rate` ∈ {0,05 ; 0,1}, `max_depth` ∈ {3, 5}, `subsample` ∈ {0,8 ; 1}, `scale_pos_weight` ∈ {1 ; bons / défauts} | 32 |

- **Raison :** chaque grille croise un réglage de complexité (profondeur,
  régularisation, nombre d'arbres) avec la pondération des classes (M03).
  Le nombre d'arbres de Random Forest est fixé à 300 : en ajouter ne cause
  pas de surapprentissage, seulement un coût en temps.
- **Temps estimé** à partir de mesures sur le PC de développement (12 cœurs
  logiques) : 7 à 15 minutes pour les 88 combinaisons × 5 plis.
  **Temps réel :** régression logistique 10 s, arbre 8 s, Random Forest
  749 s (12,5 min, au-dessus de l'estimation de 5 à 10 min), XGBoost 74 s ;
  environ 14 minutes au total.
- **Parallélisation :** GridSearchCV répartit les entraînements sur tous les
  cœurs ; Random Forest et XGBoost utilisent un seul cœur chacun pour ne pas
  se concurrencer.
- **Alternative écartée :** Random Forest allégé sans `max_features = 0,5`
  (1 à 2 minutes au lieu de 5 à 10, mais une piste de réglage en moins).

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
- **Critère retenu :** F1 maximal de la classe « défaut » (meilleur équilibre
  entre précision et rappel, cohérent avec la métrique F1 demandée). Le seuil
  du KS est calculé et affiché à côté, avec la précision et le rappel obtenus
  à chacun des deux seuils, pour comparaison.
- **Critères écartés :** rappel minimal imposé (chiffre à fixer par le
  métier) ; coût métier (coûts d'erreur non fournis par le jeu).

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
- **Tailles mesurées** (joblib, compression de niveau 3, scikit-learn 1.9.1,
  XGBoost 3.2.0) :

  | Fichier | Taille |
  |---|---|
  | `logreg.joblib` | 0,003 Mo |
  | `arbre.joblib` | 0,011 Mo |
  | `random_forest.joblib` | **17,2 Mo** (300 arbres sans limite de profondeur) |
  | `xgboost.joblib` | 0,046 Mo |
  | `resultats.json` + 4 `grille_*.parquet` | 0,05 Mo au total |

  Total : environ 17,3 Mo, dont 99,6 % pour Random Forest. GitHub refuse les
  fichiers de plus de 100 Mo et avertit au-delà de 50 Mo : les quatre modèles
  pourraient être commités. Un fichier binaire de 17 Mo alourdit cependant
  l'historique à chaque nouvel entraînement.
- **À trancher au déploiement :** commiter les modèles, les réentraîner à la
  construction de l'image Docker, ou les héberger ailleurs.

### A06. Dépôt GitHub privé jusqu'à la vérification finale — retenu
- **Raison :** l'historique est contrôlé (secrets, données personnelles) avant
  publication.

### A07. Licence MIT pour le code — retenu
- **Raison :** licence simple et permissive. Le jeu de données reste sous sa
  propre licence CC BY 4.0, citée dans le README.

### A08. Dépendances à versions exactes, séparées en deux fichiers — retenu
- **Choix :** `requirements.txt` (application, versions exactes `==`) et
  `requirements-dev.txt` (tests et notebook, inclut le premier).
- **Raison :** un modèle sauvegardé avec une version de scikit-learn peut ne
  pas se recharger avec une autre ; l'application déployée doit utiliser les
  mêmes versions que l'environnement d'entraînement. Les outils de
  développement n'ont pas à être installés au déploiement.
- **Alternatives écartées :** versions non fixées (résultats non
  reproductibles) ; Poetry ou uv (Streamlit Community Cloud lit
  `requirements.txt` directement).

### A10. Horodatages du notebook retirés automatiquement — retenu
- **Choix :** un filtre Git « clean » (`.gitattributes` + réglage local
  `git config`) passe chaque `.ipynb` dans `scripts/nettoyer_notebook.py` à
  chaque `git add` ; seule la clé `execution` (heures d'exécution des
  cellules) est retirée. Le réglage `required` fait échouer `git add` si le
  filtre ne peut pas s'exécuter.
- **Raison :** ces horodatages changent à chaque réexécution sans rien
  apporter (bruit dans l'historique) et indiqueraient publiquement les heures
  de travail. Vérifié : mêmes sources, sorties et images (empreinte
  identique), format valide, noyau conservé.
- **Limite :** le réglage `git config` est local ; il faut le refaire après un
  clonage (commande dans le README). Pour un fichier inchangé, Git ne relance
  pas le filtre : `git add --renormalize` le force.
- **Alternatives écartées :** les laisser (sans danger mais bruyant) ;
  nettoyer à la main avant chaque commit (oubli possible) ; outil externe
  (nbstripout, dépendance supplémentaire).

### A09. Installation du paquet `credit_scoring` sur Streamlit Community Cloud — à tester au déploiement
- **Choix :** le paquet est décrit dans `pyproject.toml` et installé par la
  ligne `-e .` à la fin de `requirements.txt`. Streamlit Community Cloud
  installe `requirements.txt` depuis la racine du dépôt, donc `.` désigne le
  projet lui-même.
- **Raison :** un seul fichier lu par Streamlit Cloud installe à la fois les
  bibliothèques et notre code ; localement, `pip install -r requirements-dev.txt`
  fait de même (vérifié : import réussi depuis un autre dossier).
- **Solutions de repli, dans l'ordre :** remplacer `-e .` par `.`
  (installation non modifiable) ; en dernier recours, ajouter `src/` au
  chemin d'import au début de l'application.
- **Alternative écartée :** modifier `sys.path` dans chaque fichier (fragile,
  dépend du dossier de lancement).
