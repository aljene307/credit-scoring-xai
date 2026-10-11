"""Retire les horodatages d'exécution des cellules d'un notebook Jupyter.

Deux usages :
- filtre Git « clean » (configuré dans .gitattributes et git config) : lit le
  notebook sur l'entrée standard et écrit la version nettoyée sur la sortie
  standard ; Git l'applique automatiquement à chaque `git add` d'un .ipynb ;
- à la main : `python scripts/nettoyer_notebook.py notebooks/analyse_credit.ipynb`
  nettoie le fichier sur place.

Seule la clé « execution » des métadonnées de cellule est retirée : les sources,
les sorties et les images ne changent pas.
"""

import json
import sys


def clean(text: str) -> str:
    notebook = json.loads(text)
    for cell in notebook.get("cells", []):
        cell.get("metadata", {}).pop("execution", None)
    # Même mise en forme que nbformat : clés triées, indentation de 1, UTF-8.
    return json.dumps(notebook, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            with open(path, encoding="utf-8") as f:
                cleaned = clean(f.read())
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(cleaned)
    else:
        # Octets bruts : évite la conversion des fins de ligne par Windows.
        sys.stdout.buffer.write(clean(sys.stdin.buffer.read().decode("utf-8")).encode("utf-8"))


if __name__ == "__main__":
    main()
