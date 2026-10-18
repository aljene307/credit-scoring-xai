"""Forme des données échangées avec l'API : ce que l'on envoie et ce que l'on reçoit.

Pydantic vérifie chaque demande : une valeur absente, mal typée ou hors des valeurs
autorisées est refusée (code 422) avec un message qui explique pourquoi.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

ModeleNom = Literal["complet", "sans_sensibles"]
Statut = Annotated[int, Field(ge=-2, le=8, description="Statut de remboursement : -2 pas d'utilisation, "
                                                        "-1 payé en totalité, 0 minimum payé, 1 à 8 = mois de retard")]
Facture = Annotated[int, Field(ge=-10_000_000, le=100_000_000,
                               description="Montant de la facture (NT$), négatif si trop-perçu")]
Paiement = Annotated[int, Field(ge=0, le=100_000_000, description="Montant payé (NT$)")]

EXEMPLE_CLIENT = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 22,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 16726, "BILL_AMT2": 19143, "BILL_AMT3": 18391, "BILL_AMT4": 18562, "BILL_AMT5": 11798,
    "BILL_AMT6": 0, "PAY_AMT1": 3000, "PAY_AMT2": 0, "PAY_AMT3": 371, "PAY_AMT4": 1236, "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}


class Client(BaseModel):
    """Les 23 variables d'un client, avec les codes du jeu Taiwan après recodage."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXEMPLE_CLIENT]})

    LIMIT_BAL: int = Field(gt=0, le=10_000_000, description="Plafond de crédit (NT$)")
    SEX: Literal[1, 2] = Field(description="1 = homme, 2 = femme")
    EDUCATION: Literal[1, 2, 3, 4] = Field(description="1 = 3e cycle, 2 = université, 3 = lycée, 4 = autres ou inconnu")
    MARRIAGE: Literal[1, 2, 3] = Field(description="1 = marié, 2 = célibataire, 3 = autres")
    AGE: int = Field(ge=18, le=100, description="Âge en années")
    PAY_0: Statut
    PAY_2: Statut
    PAY_3: Statut
    PAY_4: Statut
    PAY_5: Statut
    PAY_6: Statut
    BILL_AMT1: Facture
    BILL_AMT2: Facture
    BILL_AMT3: Facture
    BILL_AMT4: Facture
    BILL_AMT5: Facture
    BILL_AMT6: Facture
    PAY_AMT1: Paiement
    PAY_AMT2: Paiement
    PAY_AMT3: Paiement
    PAY_AMT4: Paiement
    PAY_AMT5: Paiement
    PAY_AMT6: Paiement


class DemandePrediction(BaseModel):
    client: Client
    modele: ModeleNom = Field("sans_sensibles", description="Modèle utilisé ; par défaut, celui recommandé au "
                                                             "déploiement (sans variables sensibles)")
    seuil: float | None = Field(None, gt=0, lt=1, description="Seuil de décision ; par défaut, le seuil recommandé")


class DemandeExplication(BaseModel):
    client: Client
    modele: ModeleNom = Field("sans_sensibles", description="Modèle expliqué")
    methodes: list[Literal["shap", "lime"]] = Field(["shap", "lime"], min_length=1)
    nombre_variables: int = Field(5, ge=1, le=23, description="Nombre de variables principales renvoyées")


class Prediction(BaseModel):
    modele: ModeleNom
    probabilite: float = Field(description="Probabilité de défaut le mois suivant (entre 0 et 1)")
    seuil: float
    signale: bool = Field(description="Vrai si la probabilité atteint le seuil")
    decision: Literal["signalé comme risqué", "non signalé"]
    variables_ignorees: list[str] = Field(description="Variables reçues mais non utilisées par ce modèle")
    duree_ms: float


class RegleLime(BaseModel):
    regle: str
    poids: float


class Accord(BaseModel):
    communes: int = Field(description="Variables communes aux principales de SHAP et de LIME")
    meme_signe: int
    variables: list[str]


class Explication(Prediction):
    echelle: Literal["probabilité"] = "probabilité"
    valeur_de_base: float | None = Field(None, description="Probabilité moyenne des clients de référence")
    contributions_shap: dict[str, float] | None = Field(
        None, description="Contribution de chaque variable (valeur de base + somme = probabilité)")
    principales_shap: list[str] | None = None
    regles_lime: list[RegleLime] | None = None
    poids_lime: dict[str, float] | None = Field(None, description="Poids LIME par variable")
    accord: Accord | None = Field(None, description="Présent si SHAP et LIME sont demandés")
    phrase: str | None = Field(None, description="Synthèse en langage courant (nécessite SHAP)")


class VariableAttendue(BaseModel):
    description: str
    utilisee: bool


class InformationsModele(BaseModel):
    modele: ModeleNom
    nom: str
    variables: dict[str, VariableAttendue]
    variables_ignorees: list[str]
    seuil_recommande: float
    methode_seuil: str
    auc_validation_croisee: float
    auc_test: float
    clients_test: int


class Sante(BaseModel):
    statut: Literal["ok"]
    modeles_disponibles: list[ModeleNom]
    versions: dict[str, str]


class ExplicationsGlobales(BaseModel):
    modele: Literal["complet"]
    echelle: str
    valeur_de_base: float
    clients_expliques: int
    contribution_moyenne: dict[str, float] = Field(description="Moyenne des |valeurs SHAP| par variable")
    permutation: dict[str, dict[str, float]] = Field(description="Baisse d'AUC moyenne, par modèle puis variable")
