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
- **Installation :** Matplotlib était dans `requirements-dev.txt` tant que seul
  le notebook l'utilisait ; il est passé dans `requirements.txt` avec SHAP
  (étape 5).
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

### M03. Pondération des classes : testée, puis retirée des modèles finaux — retenu
- **Premier choix (testé) :** `class_weight ∈ {None, "balanced"}` (scikit-learn)
  et `scale_pos_weight ∈ {1, n_négatifs / n_positifs}` (XGBoost) dans la grille ;
  GridSearchCV décide.
- **Décision finale :** les quatre modèles finaux sont entraînés **sans
  pondération** (grilles de M06 sans ces hyperparamètres).
- **Raison de la décision finale :** la pondération n'apporte rien de mesurable
  sur les meilleures combinaisons (au plus +0,0004 d'AUC, contre 0,005 à 0,011
  de variation d'un pli à l'autre) et elle fausse les probabilités (elles ne
  sont plus des probabilités de défaut).
- **Preuve (premier GridSearch, validation croisée sur l'entraînement) :**

  | Modèle | Meilleure AUC avec pondération | Meilleure AUC sans | Écart |
  |---|---|---|---|
  | Régression logistique | 0,7276 | 0,7272 | +0,0004 |
  | Arbre de décision | 0,7598 | 0,7665 | −0,0066 |
  | Random Forest | 0,7849 | 0,7845 | +0,0004 |
  | XGBoost | 0,7829 | 0,7828 | +0,0001 |

- **Exception à connaître :** pour un arbre de profondeur 3, la pondération
  améliore nettement l'AUC (0,7486 contre 0,7328, soit +0,016) : avec 8 feuilles
  au plus, elle change les questions posées par l'arbre, pas seulement les
  probabilités. Cela ne concerne pas les modèles finaux, mais compte pour le
  coût d'un arbre lisible (M08).
- **Effet à connaître :** la pondération gonfle les probabilités prédites
  (moyenne proche de 50 % au lieu de 22 %). Elle change peu le classement des
  clients, donc peu l'AUC, mais les probabilités ne sont plus des
  probabilités de défaut réelles.
- **Alternative écartée :** SMOTE (crée des clients synthétiques, dépendance
  supplémentaire, explications SHAP sur des données artificielles).
- **Résultat du premier GridSearch (avant la décision finale) :** GridSearchCV
  avait retenu la pondération pour la régression logistique, Random Forest et
  XGBoost, mais pour un gain d'AUC négligeable (+0,0004, +0,0004 et +0,0001) ;
  pour l'arbre, la meilleure combinaison était sans pondération (+0,0066). Ce
  constat a conduit à la décision finale ci-dessus.

### M04. Modèles de référence entraînés hors ligne — retenu
- **Choix :** un script entraîne et sauvegarde les modèles ; l'interface permet
  aussi de relancer un GridSearch avec des grilles réduites.
- **Raison :** un GridSearch complet sur 30 000 lignes prend plusieurs minutes
  et dépasse les ressources de Streamlit Community Cloud.
- **Alternative écartée :** tout entraîner à chaque ouverture de l'application.

### M05. Relation non linéaire entre PAY_0 et le risque — confirmé (étapes 3 et 5)
- **Confirmation par SHAP (1 000 clients du test) :** contribution moyenne de
  PAY_0 pour XGBoost : −3,8 / −3,4 / −5,1 points pour −2 / −1 / 0 (plat), +3,7
  à 1, **+33,1 à 2** (saut de 29 points). La régression logistique ajoute
  exactement +0,588 log-odds à chaque cran : elle juge −2 bien plus sûr que 0
  (−1,18 contre −0,01) alors que leurs taux de défaut observés sont presque
  égaux (13,2 % et 12,8 %), et sous-estime le saut à 2 mois.
- **Résultat :** AUC de validation 0,727 pour la régression logistique contre
  0,771 à 0,786 pour les trois modèles à base d'arbres (grilles finales). Le
  meilleur `C` est la plus grande valeur de la grille (régularisation la plus
  faible) et l'AUC varie de moins de 0,002 entre `C` = 0,1 et 10 : le modèle
  n'est pas freiné par la régularisation mais limité par sa forme linéaire. Confirmation à
  chercher dans les graphiques de dépendance SHAP (étape 5).
- **Constat :** le taux de défaut est plat de PAY_0 = -2 à 0 (13 à 17 %), puis
  saute à 1 (33,9 %) et à 2 (69,1 %).
- **Conséquence attendue :** la régression logistique suppose un effet
  linéaire de PAY_0 sur le logit du risque ; elle peut donc être désavantagée
  par rapport aux arbres, qui découpent librement les seuils.
- **À vérifier à l'étape 3 :** comparer les performances, et regarder la forme
  de l'effet de PAY_0 dans les graphiques de dépendance SHAP.

### M06. Grilles d'hyperparamètres — retenu
**Grilles finales** (sans pondération, M03) :

| Modèle | Grille | Combinaisons |
|---|---|---|
| Régression logistique | `C` ∈ {0,01 ; 0,1 ; 1 ; 10} | 4 |
| Arbre de décision | `max_depth` ∈ {3, 5, 8, 12, 16, 20}, `min_samples_leaf` ∈ {1, 20, 50, 100, 200, 400, 800, 1600} | 48 |
| Random Forest (300 arbres) | `max_depth` ∈ {8, 12, sans limite}, `min_samples_leaf` ∈ {1, 10}, `max_features` ∈ {sqrt, 0,5} | 12 |
| XGBoost | `n_estimators` ∈ {300, 500, 800}, `learning_rate` ∈ {0,01 ; 0,02 ; 0,03}, `max_depth` ∈ {3, 4, 5}, `subsample` ∈ {0,4 ; 0,6 ; 0,8} | 81 |

- **Raison :** chaque grille fait varier la complexité du modèle (profondeur,
  régularisation, nombre d'arbres, vitesse d'apprentissage). Le nombre
  d'arbres de Random Forest est fixé à 300 : en ajouter ne cause pas de
  surapprentissage, seulement un coût en temps.
- **Historique :**
  1. Première grille (avec pondération, 88 combinaisons) : plusieurs
     meilleures valeurs en bord de grille (arbre : profondeur 12 ;
     XGBoost : 200 arbres, taux 0,05, profondeur 3).
  2. Grilles de l'arbre et de XGBoost prolongées (sans pondération) : l'arbre
     reste en bord (`min_samples_leaf` = 400, l'AUC monte encore de 0,7691 à
     0,7707 entre 200 et 400) ; XGBoost a ses 4 réglages en bord.
  3. Second prolongement : l'arbre n'a plus aucun bord (profondeur 16 entre
     12 et 20 ; feuille 400 entre 200 et 800). XGBoost reste en bord sur 3
     réglages (taux 0,01, profondeur 5, sous-échantillon 0,4).
- **Bords restants, mesurés :**
  - XGBoost : surface presque plate, l'optimum glisse le long d'une crête.
    Écart entre la valeur au bord et sa voisine : 0,0003 (taux), 0,0003
    (profondeur), 0,0006 (sous-échantillon) ; les 10 meilleures combinaisons
    tiennent en 0,0009, pour un écart-type entre plis de 0,0054.
  - **Décision pour XGBoost : arrêt après le second prolongement.** Pousser
    plus loin ferait gagner moins que le bruit entre plis, et choisir le
    meilleur parmi des centaines de combinaisons presque égales rendrait le
    score de validation légèrement trop optimiste. Limite citée telle quelle.
    *Alternative écartée :* troisième prolongement (taux 0,005, profondeur 6
    et 7, sous-échantillon 0,2 et 0,3), gain attendu inférieur à 0,001.
  - Régression logistique : `C` = 10 ; AUC 0,7272 à `C` = 1 et à `C` = 10 (plat).
  - Random Forest : profondeur sans limite et feuille 10 ; écarts de 0,001 et
    0,0018 avec les valeurs voisines.
- **Parallélisation :** GridSearchCV répartit les entraînements sur tous les
  cœurs ; Random Forest et XGBoost utilisent un seul cœur chacun pour ne pas
  se concurrencer.
- **Temps réels** (PC de développement, 12 cœurs logiques) : régression
  logistique 6 s, arbre 17 s, Random Forest 260 s, XGBoost 242 s. La première
  estimation pour Random Forest (5 à 10 min avec 24 combinaisons) avait été
  dépassée (12,5 min).
- **Alternative écartée :** Random Forest allégé sans `max_features = 0,5`
  (une piste de réglage en moins).

### M07. Modèle de référence : XGBoost — retenu (choisi avant le test)
- **Choix :** XGBoost est le modèle de référence de l'application et de
  l'API. Il est choisi sur la validation croisée de l'entraînement, **avant**
  toute utilisation du jeu de test, pour que l'évaluation finale ne serve pas
  à choisir.
- **Raisons :**
  - meilleure AUC de validation (0,786, Gini 0,572), à égalité avec Random
    Forest (0,785) : l'écart est 10 fois plus petit que la variation entre
    plis ;
  - fichier 30 fois plus léger que Random Forest (0,40 Mo contre 12,3 Mo) :
    déploiement et chargement plus simples ;
  - surapprentissage bien plus faible (AUC entraînement − validation : 0,045
    contre 0,147 pour Random Forest) ;
  - probabilités cohérentes avec le taux réel (moyenne hors pli 22,05 % pour
    22,12 % de défauts) ;
  - explications SHAP exactes et rapides pour les arbres de XGBoost.
- **Alternatives écartées :** Random Forest (même performance, fichier lourd,
  forte mémorisation) ; régression logistique (la plus interprétable, mais
  0,059 d'AUC en moins) ; arbre de décision (lisible seulement à faible
  profondeur, voir M08).

### M09. GridSearch relancé depuis l'interface : résultats gardés dans la session — retenu
- **Choix :** la page « 3. Modélisation » affiche les modèles de référence
  (fichiers de `models/`) et permet de relancer un GridSearch sur la
  préparation de la page 2, avec des valeurs choisies dans les grilles de
  référence. Le nombre de combinaisons et la durée estimée (proportionnelle au
  GridSearch de référence) sont affichés avant le lancement. Les modèles ainsi
  obtenus restent dans la session et ne remplacent pas ceux de référence.
- **Raison :** le sujet demande de choisir modèles et hyperparamètres depuis
  l'interface ; les modèles de référence doivent rester stables pour
  l'évaluation, l'API et le one-pager. Par défaut, seules les meilleures
  valeurs de référence sont cochées (une combinaison, quelques secondes).
- **Alternatives écartées :** écraser les modèles de référence depuis
  l'interface ; saisie libre des valeurs (risque de valeurs invalides et de
  durées imprévisibles).

### M08. Coût d'un arbre lisible (profondeur 3) — constat
- **Mesure (validation croisée) :** meilleur arbre de profondeur 3 sans
  pondération : AUC 0,734 (Gini 0,468) ; meilleur arbre toutes profondeurs :
  0,771 ; XGBoost : 0,786 (Gini 0,572).
- **Coût de la lisibilité :** −0,052 d'AUC par rapport à XGBoost, −0,037 par
  rapport au meilleur arbre. Avec pondération (M03), l'arbre de profondeur 3
  atteint 0,749 : coût réduit à −0,037 par rapport à XGBoost.
- **Usage :** argument chiffré pour le compromis performance / interprétabilité
  demandé dans le one-pager.

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
- **Curseur de la page Évaluation :** pas de 0,001 ; sur la position du seuil
  recommandé, la valeur exacte est appliquée (0,2685 pour XGBoost, et non son
  arrondi 0,27, qui signalait 6 clients de moins). Le premier affichage
  correspond ainsi exactement au tableau des métriques.

### E03. Courbe de calibration du modèle de référence — retenu
- **Choix :** tracer, pour XGBoost, le taux de défaut observé en fonction de
  la probabilité prédite, sur **10 tranches de même effectif** (600 clients du
  test chacune), avec la diagonale « probabilité = taux observé » et le score
  de Brier. Aucune correction sans accord explicite.
- **Raison :** l'AUC ne dit rien de la justesse des probabilités ; le décideur
  lira « 35 % de risque » comme un vrai risque. La moyenne hors pli (22,05 %
  pour 22,12 % observés) ne vérifie la calibration qu'en moyenne, pas tranche
  par tranche. Des tranches de même effectif donnent des points stables.
- **Alternatives écartées :** se contenter de la moyenne des probabilités ;
  tranches de même largeur (quelques dizaines de clients dans les tranches
  élevées) ; recalibration automatique (modification du modèle, à décider
  séparément).
- **Résultat (test) :** écart moyen de 0,9 point entre probabilité moyenne et
  taux observé, écart maximal de 2,4 points (tranche 5 : 13,6 % prédit,
  16,0 % observé), soit au plus 1,8 écart-type d'échantillonnage : compatible
  avec le hasard. Score de Brier 0,135, contre 0,172 pour un modèle naïf qui
  prédirait 22,12 % pour tous (−21,5 %). **Aucune recalibration nécessaire.**

### E04. Les quatre modèles évalués sur le test — retenu
- **Choix :** les quatre modèles de référence sont évalués sur les 6 000
  clients du test, chacun à son seuil F1 fixé à l'étape 3.
- **Raison :** le modèle de référence étant déjà choisi (M07), le test ne sert
  pas à choisir ; les quatre scores vérifient que le classement de la
  validation croisée se confirme et permettent les courbes superposées.
- **Alternative écartée :** évaluer XGBoost seul.
- **Résultat :** AUC test 0,779 (XGBoost), 0,774 (Random Forest), 0,765
  (arbre), 0,710 (régression logistique) ; même classement qu'en validation
  croisée, avec une baisse de 0,005 à 0,017, chaque valeur de validation
  croisée restant dans l'intervalle de confiance du test.

### E05. Intervalles de confiance par bootstrap apparié — retenu
- **Choix :** 1 000 tirages avec remise des 6 000 clients du test ; les mêmes
  tirages pour les quatre modèles (bootstrap apparié), ce qui donne aussi un
  intervalle pour l'écart d'AUC entre deux modèles.
- **Raison :** 6 000 clients ne sont qu'un échantillon ; sans intervalle, on
  risque d'interpréter un écart de quelques millièmes comme réel.
- **Alternative écartée :** pas d'intervalle.
- **Résultat :** XGBoost 0,779 [0,765 ; 0,795]. Écart XGBoost − Random Forest :
  +0,005 [−0,000 ; +0,009], l'intervalle contient 0 (égalité confirmée) ;
  XGBoost − arbre : +0,014 [+0,007 ; +0,020] ; XGBoost − régression
  logistique : +0,069 [+0,056 ; +0,082].

### E06. Phrase de synthèse pour le décideur sous la matrice de confusion — retenu
- **Choix :** sous la matrice de confusion, une phrase mise à jour avec le
  curseur donne le nombre de clients signalés, le nombre de vrais défauts
  parmi eux et le nombre de défauts manqués.
- **Raison :** un décideur non spécialiste lit des effectifs plus facilement
  qu'une précision ou un rappel.
- **Exemple (XGBoost, seuil F1) :** « Sur 6 000 clients, le modèle en signale
  1 423 comme risqués : 743 d'entre eux font réellement défaut (52 %). Il
  manque 584 défauts sur 1 327. »

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

**Mesures de l'étape 5 :** 1 000 clients du test tirés avec la graine 42 sauf
mention contraire ; PC de développement (12 cœurs logiques) ; shap 0.51.0,
lime 0.2.0.1.

### X02. Échelle des valeurs SHAP : probabilité pour XGBoost, log-odds pour la régression logistique — retenu
- **Choix :** XGBoost (et les autres modèles à arbres) : TreeExplainer
  « interventionnel » en **points de probabilité**. Régression logistique :
  LinearExplainer en **log-odds**, l'échelle interne du modèle.
- **Mesures :**

  | Méthode | Durée pour 1 000 clients | Somme = sortie du modèle |
  |---|---|---|
  | XGBoost, probabilité (TreeExplainer, 500 clients de référence) | 158 s | oui (écart 4 × 10⁻⁷) |
  | XGBoost, log-odds (TreeExplainer natif) | 4 s | oui |
  | Régression logistique, log-odds (LinearExplainer) | 0,1 s | oui (10⁻¹⁴) |
  | Méthode commune en probabilité (PermutationExplainer, 23 variables) | 38 min (XGBoost), 11 min (rég. log.) | oui |

  La permutation est **bruitée** : deux calculs du même client avec 500 et
  5 000 évaluations diffèrent jusqu'à 1,1 point ; comparée à la valeur exacte
  de XGBoost, elle s'en écarte jusqu'à 4,7 points, et les 5 premières
  variables ne coïncident que pour 53 % des clients, sans amélioration avec
  plus d'évaluations.
- **Raison :** le décideur lit « +31 points de risque » ; aucune méthode exacte
  n'existe en probabilité pour la régression logistique, et la seule méthode
  commune mesurée est lente et bruitée.
- **Conséquence :** les valeurs SHAP des deux modèles ne sont pas sur la même
  échelle ; on compare donc les modèles par des mesures sans unité (rangs des
  variables, importance par permutation, recouvrement des 5 premières).
- **Alternatives écartées :** log-odds pour les deux (même échelle exacte, mais
  illisible pour un non-spécialiste) ; permutation en probabilité pour les deux
  (lente et bruitée).

### X03. Échantillon de référence : 500 clients stratifiés sur le score — retenu
- **Choix :** 500 clients de l'entraînement, 50 par décile de probabilité
  prédite (tirage stratifié, graine 42), passés à SHAP par
  `shap.maskers.Independent(..., max_samples=500)`.
- **Défaut de SHAP découvert :** passé directement à `TreeExplainer`,
  l'échantillon est **réduit en silence à 100 clients tirés au hasard**, quelle
  que soit sa taille (vérifié : 100 lignes conservées sur 500). Sans le masker,
  la valeur de base variait de 17,8 % à 25,3 % et la stratification n'avait
  aucun effet. Un test vérifie désormais que la valeur de base est bien la
  probabilité moyenne des clients de référence fournis.
- **Mesures (après correction, 100 clients expliqués, 3 tirages) :**

  | Référence | Valeur de base | Écart max entre tirages | Durée / 1 000 clients |
  |---|---|---|---|
  | 200, simple | 20,2 à 25,5 % | 4,8 points | 48 s |
  | 200, stratifié | 22,0 à 22,6 % | 1,5 point | 48 s |
  | 500, simple | 21,5 à 22,8 % | 1,1 point | — (durée aberrante, non retenue) |
  | **500, stratifié** | **22,04 à 22,07 %** | **0,4 point** | 59 à 158 s |
  | 1 000, stratifié | 22,14 à 22,18 % | 0,4 point | 331 s |

  Probabilité moyenne réelle sur l'entraînement : 22,08 %.
- **Raison :** la valeur de base est la référence de toute explication (« 71 %
  contre 22 % en moyenne ») ; elle doit valoir la vraie moyenne. 1 000 clients
  n'apportent rien de plus pour un temps 2 à 5 fois plus long.
- **Alternatives écartées :** tirage simple (base fausse de plusieurs
  points selon le tirage) ; 1 000 clients (plus lent sans gain).

### X04. Explications globales sur 1 000 clients du test, calculées hors ligne — retenu
- **Mesure :** importance SHAP moyenne sur 1 000 clients contre les 6 000 :
  corrélation des rangs 0,999, **mêmes 10 premières variables dans le même
  ordre** ; durée 158 s contre 916 s.
- **Raison :** même information pour 6 fois moins de temps. 158 s restent trop
  longs pour un affichage dans l'application : les valeurs globales sont
  calculées une fois hors ligne, comme les modèles (M04).
- **Alternative écartée :** les 6 000 clients (15 minutes pour un classement
  identique).

### X05. SHAP et LIME pour XGBoost et la régression logistique — retenu
- **Choix :** SHAP et LIME pour XGBoost (référence) et la régression
  logistique (modèle le plus interprétable, pour la comparaison) ; importance
  par permutation et importance propre au modèle pour les quatre modèles.
- **Mesure :** SHAP sur Random Forest : 6,2 s pour 10 clients avec seulement
  100 clients de référence (mesure faite avant la correction de X03), soit au
  moins 10 minutes pour 1 000 clients, et davantage avec 500 clients de
  référence.
- **Alternative écartée :** SHAP sur les quatre modèles.

### X06. Importance par permutation : sur le test, baisse d'AUC, 10 répétitions — retenu
- **Mesures (XGBoost) :** 22 s sur le test. Baisse d'AUC : PAY_0 0,079
  (± 0,005), LIMIT_BAL 0,018, BILL_AMT1 0,015, puis de 0,003 à 0,005 pour
  PAY_2, PAY_AMT2, PAY_3, PAY_AMT3, PAY_AMT1. Sur l'entraînement, les baisses
  sont plus fortes (somme 0,217 contre 0,150 sur le test ; AGE 0,0056 contre
  0,0008) : elles mesurent aussi ce que le modèle a mémorisé. Corrélation des
  rangs entraînement / test : 0,76. Avec une autre graine, l'ordre des 4e et
  5e variables change (écarts inférieurs à un écart-type).
- **Raison :** le test mesure la dépendance du modèle sur des clients
  nouveaux. Les variables proches doivent être lues avec leur écart-type, pas
  par leur rang strict.
- **Autres modèles (test, 10 répétitions) :** régression logistique 2 s
  (PAY_0 0,111), arbre 2 s (PAY_0 0,081, PAY_2 0,037), Random Forest 24 s
  (PAY_0 0,071).
- **Alternatives écartées :** sur l'entraînement (gonfle les variables
  mémorisées) ; 5 répétitions (écarts-types moins précis pour le même ordre
  instable des variables proches).

### X07. LIME : 10 000 voisins, découpage en quartiles, graine fixe — retenu
- **Mesures de stabilité (3 graines) :**
  - client 1358 : 1 000 voisins → les 5 premières changent d'une graine à
    l'autre ; dès 5 000 voisins → mêmes variables, mêmes signes, même ordre ;
  - 20 clients du test : 5 premières strictement identiques pour 3/20 (5 000
    voisins), 5/20 (10 000), 7/20 (25 000) ; variables communes aux 3
    graines : 3,5 / 3,8 / 4,2 sur 5 en moyenne.
  - Durée : 0,1 s (5 000), 0,4 s (10 000), 0,8 s (25 000) par explication.
- **Raison :** 10 000 voisins rendent l'explication assez stable (3,8 variables
  sur 5 communes entre graines) pour un affichage interactif ; la graine fixe
  rend l'explication reproductible. **Limite à citer :** même à 25 000
  voisins, LIME reste instable pour une partie des clients.
- **Alternatives écartées :** 5 000 voisins (moins stable) ; 25 000 voisins
  (gain modeste pour un temps doublé).

### X08. Comparaison SHAP / LIME : recouvrement des 5 premières et accord des signes — retenu
- **Mesures (20 clients du test, LIME à 10 000 voisins) :**
  - XGBoost : 3,8 variables communes sur 5 en moyenne (de 2 à 5), signes
    concordants 72 sur 75 ; variables communes les plus fréquentes : PAY_0
    (20 clients sur 20), LIMIT_BAL (17), BILL_AMT1 (10).
  - Régression logistique : 3,6 sur 5 (de 3 à 5), signes concordants 72 sur
    72 ; MARRIAGE apparaît chez 14 clients sur 20, ce qui renvoie à
    l'analyse d'équité.
  - Client 1358 : 5 variables communes sur 5, toutes de même signe.
- **Raison :** des mesures simples à expliquer qui ne dépendent pas de
  l'échelle (LIME n'est pas en points de probabilité).

### X09. Équité : variables relais, modèle sans variables sensibles, erreurs par groupe — retenu
- **Variables relais** (prédire chaque variable sensible à partir des 19
  autres, validation croisée sur l'entraînement) : SEX AUC 0,598 ;
  MARRIAGE 55,1 % de bonnes réponses contre 53,4 % en prédisant toujours la
  modalité la plus fréquente ; EDUCATION 53,0 % contre 46,9 % (AUC 0,638 :
  le plafond médian est de 200 000 NT$ en 3e cycle contre 80 000 au lycée) ;
  AGE R² 0,036 (erreur moyenne 7,4 ans contre 7,5 sans modèle). Corrélation
  de rang AGE / LIMIT_BAL : 0,19. Les autres variables ne permettent de
  retrouver que faiblement les variables sensibles, un peu mieux pour le
  niveau d'études.
- **XGBoost sans les 4 variables sensibles** (mêmes hyperparamètres) : AUC de
  validation 0,7841 ± 0,0051, contre 0,7860 avec : −0,002, en dessous de la
  variation entre plis.
- **Erreurs sur le test au seuil recommandé** (tests du khi-deux) :

  | Groupe | Clients | Taux de défaut | Rappel | Faux positifs |
  |---|---|---|---|---|
  | Femmes | 3 598 | 21,3 % | 54,8 % | 12,7 % |
  | Hommes | 2 402 | 23,4 % | 57,6 % | 17,4 % |
  | 21-25 ans | 767 | 25,6 % | 66,3 % | 21,2 % |
  | 26-35 ans | 2 492 | 19,7 % | 51,4 % | 12,0 % |
  | 36-45 ans | 1 775 | 22,4 % | 53,4 % | 13,5 % |
  | 46-55 ans | 788 | 25,1 % | 60,1 % | 17,3 % |
  | 56 ans et + | 178 | 25,8 % | 65,2 % | 22,7 % |

  Rappel selon le sexe : p = 0,35 (écart non significatif). Faux positifs
  selon le sexe : p < 0,0001 (un bon payeur homme est plus souvent signalé
  qu'une bonne payeuse). Rappel et faux positifs selon l'âge : p = 0,002 et
  p < 0,0001 (les 21-25 ans et les 56 ans et plus sont plus souvent signalés).
- **Sans les variables sensibles :** l'écart de faux positifs entre sexes se
  réduit (15,8 % contre 12,4 %, au lieu de 17,4 % contre 12,7 %) mais reste
  significatif (p = 0,001) ; les écarts selon l'âge persistent (p < 0,0001).
- **Lecture :** les groupes les plus signalés sont aussi ceux dont le taux de
  défaut réel est le plus élevé, et le modèle est calibré dans chaque groupe
  (probabilité moyenne égale au taux observé). Avec des taux de défaut
  différents, un modèle calibré ne peut pas avoir le même taux de faux
  positifs dans chaque groupe : il faut choisir le critère d'équité visé.
- **Recommandation pour le one-pager (décision de l'auteur) :** déployer la
  version de XGBoost **sans les variables sensibles** (SEX, AGE, MARRIAGE,
  EDUCATION), qui ne coûte que −0,002 d'AUC en validation croisée (0,7841
  contre 0,7860, moins que la variation entre plis) et réduit l'écart de
  faux positifs entre hommes et femmes. Le modèle complet reste dans
  l'application pour montrer l'analyse d'équité (poids des variables
  sensibles, erreurs par groupe, comparaison avec et sans).
- **Limite à citer avec cette recommandation :** retirer les variables
  sensibles ne supprime pas tous les écarts : ceux selon l'âge persistent,
  portés par d'autres variables et par des taux de défaut réellement
  différents entre groupes.

### X10. Phrase de synthèse générée à partir des 3 plus fortes contributions SHAP — retenu
- **Choix :** niveau de risque (« élevé » si signalé au seuil, « modéré » si
  au-dessus de la moyenne sans être signalé, « faible » sinon), probabilité
  comparée à la valeur de base, puis les 3 plus fortes contributions dans le
  sens de la décision, en langage courant ; les statuts de remboursement
  identiques sur plusieurs mois sont regroupés.
- **Exemple (client 1358) :** « Le risque de défaut est élevé (71 % contre
  22 % en moyenne), principalement à cause d'un retard de paiement de 2 mois
  en septembre et en août et d'un plafond de crédit faible (20 000 NT$). »
- **Alternative écartée :** lister les variables avec leurs valeurs SHAP
  (lisible seulement par un spécialiste).
- **Ordre des mois :** quand plusieurs statuts identiques sont regroupés, les
  mois sont écrits du plus récent au plus ancien (« en août et en juillet »),
  quel que soit l'ordre de leurs contributions.
- **Moyenne citée :** probabilité moyenne de l'échantillon de référence pour le
  modèle expliqué : 22 % pour XGBoost, 23 % pour la régression logistique (ce
  modèle prédit en moyenne un peu plus de risque sur les mêmes 500 clients).

### X12. Importance propre au modèle : des mesures qui s'additionnent — retenu (correction)
- **Défaut constaté :** une première version additionnait, pour chaque variable
  catégorielle, le **gain moyen** de XGBoost et les **|coefficients|** de la
  régression logistique de ses modalités. Résultat faux : EDUCATION, SEX et
  MARRIAGE passaient devant PAY_0 pour la régression logistique, et EDUCATION
  en 3e position pour XGBoost, en contradiction avec l'importance par
  permutation.
- **Preuve :** pour SEX, coefficients −0,353 (homme) et −0,473 (femme) : la
  somme des valeurs absolues donne 0,826 alors que l'écart réel entre les deux
  modalités n'est que de 0,120 (le reste est un décalage commun compensé par
  la constante). Pour XGBoost, en part du gain total : PAY_0 40,6 %,
  LIMIT_BAL 5,4 %, PAY_2 4,6 %, EDUCATION 1,5 %, MARRIAGE 0,6 %, SEX 0,5 %.
- **Choix :** XGBoost : part du gain total ; arbre et Random Forest : part de la
  réduction d'impureté (déjà additives) ; régression logistique : |coefficient|
  par écart-type pour une numérique, écart entre la modalité la plus et la
  moins risquée pour une catégorielle. Deux tests empêchent le retour du défaut.
- **Limite à citer :** pour la régression logistique, EDUCATION reste en tête
  (écart de 1,135) à cause de la modalité « Autres ou inconnu », qui ne
  concerne que 1,6 % des clients ; l'importance par permutation (0,0019)
  mesure son effet réel sur l'ensemble des clients.

### X11. Page « 5. Explicabilité » : global hors ligne, client en direct — retenu
- **Choix :** les valeurs SHAP globales (1 000 clients), l'importance par
  permutation des quatre modèles et l'analyse d'équité sont calculées une fois
  par `scripts/expliquer.py` (3 min 45 s sur le PC de développement) et lues
  par la page. L'explication d'un client (SHAP, LIME, phrase) est calculée en
  direct : client réel du test choisi par son identifiant ou au hasard, ou
  client fictif obtenu en modifiant ses valeurs dans un formulaire.
- **Raison :** 138 s pour SHAP global ne sont pas acceptables à l'affichage ;
  un client prend moins d'une seconde (SHAP environ 0,2 s, LIME environ 0,4 s).
- **Onglets :** importance globale (propre au modèle et par permutation), SHAP
  global (beeswarm), dépendance, explication d'un client (waterfall, LIME,
  accord SHAP / LIME, phrase), équité.
- **Régression logistique :** la page indique que ses valeurs sont en log-odds
  et explique ce que cela veut dire.
- **Alternative écartée :** tout calculer à l'ouverture de la page.

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
  | `arbre.joblib` | 0,006 Mo |
  | `random_forest.joblib` | **12,3 Mo** (300 arbres sans limite de profondeur) |
  | `xgboost.joblib` | 0,40 Mo (500 arbres de profondeur 5) |
  | `resultats.json` + 4 `grille_*.parquet` | environ 0,05 Mo |

  Total : environ 12,8 Mo, dont 96 % pour Random Forest (17,2 Mo avec la
  première grille). GitHub refuse les fichiers de plus de 100 Mo et avertit
  au-delà de 50 Mo : les quatre modèles pourraient être commités. Un fichier
  binaire de 12 Mo alourdit cependant l'historique à chaque nouvel
  entraînement.
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
- **Complément :** le filtre fusionne aussi les morceaux consécutifs d'une même
  sortie texte. Jupyter reçoit un `print` tantôt en un morceau, tantôt en
  deux : sans cette fusion, deux exécutions identiques donnaient deux fichiers
  différents. Le tableau des grilles n'utilise plus le `Styler` de pandas, qui
  génère des identifiants aléatoires à chaque exécution. Vérifié : deux
  exécutions successives donnent exactement la même version enregistrée, et le
  texte affiché des 106 autres cellules est inchangé.
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
