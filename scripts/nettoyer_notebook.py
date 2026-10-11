"""Retire les horodatages d'exécution des cellules d'un notebook Jupyter.

Deux usages :
- filtre Git « clean » (configuré dans .gitattributes et git config) : lit le
  notebook sur l'entrée standard et écrit la version nettoyée sur la sortie
  standard ; Git l'applique automatiquement à chaque `git add` d'un .ipynb ;
- à la main : `python scripts/nettoyer_notebook.py notebooks/analyse_credit.ipynb`
  nettoie le fichier sur place.

Deux nettoyages, sans changer ce qui est affiché :
- la clé « execution » des métadonnées de cellule (heures d'exécution) est retirée ;
- les morceaux consécutifs d'une même sortie texte (un `print` que Jupyter reçoit
  tantôt en un morceau, tantôt en deux) sont fusionnés, pour que deux exécutions
  identiques donnent exactement le même fichier.
Les sources, les images et le texte affiché ne changent pas.
"""

import json
import sys


def _as_text(value) -> str:
    return "".join(value) if isinstance(value, list) else value


def _merge_streams(outputs: list) -> list:
    """Fusionne les sorties texte consécutives de même nom (stdout ou stderr)."""
    merged = []
    for output in outputs:
        previous = merged[-1] if merged else None
        if (
            output.get("output_type") == "stream"
            and previous is not None
            and previous.get("output_type") == "stream"
            and previous.get("name") == output.get("name")
        ):
            text = _as_text(previous["text"]) + _as_text(output["text"])
            previous["text"] = text.splitlines(keepends=True)  # même découpage que nbformat
        else:
            merged.append(output)
    return merged


def clean(text: str) -> str:
    notebook = json.loads(text)
    for cell in notebook.get("cells", []):
        cell.get("metadata", {}).pop("execution", None)
        if "outputs" in cell:
            cell["outputs"] = _merge_streams(cell["outputs"])
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
