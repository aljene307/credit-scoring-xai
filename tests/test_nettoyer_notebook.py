"""Tests du script de nettoyage des horodatages du notebook."""

import importlib.util
import json
from pathlib import Path

import nbformat

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nettoyer_notebook.py"
spec = importlib.util.spec_from_file_location("nettoyer_notebook", SCRIPT)
nettoyer_notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nettoyer_notebook)


def make_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("print('é')")
    cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="é\n")]
    cell.metadata["execution"] = {"iopub.execute_input": "2026-10-11T20:03:13.589533Z"}
    cell.metadata["tags"] = ["a-garder"]
    nb.cells = [cell, nbformat.v4.new_markdown_cell("# Titre")]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def test_clean_removes_only_execution_timestamps():
    cleaned = json.loads(nettoyer_notebook.clean(nbformat.writes(make_notebook())))
    code_cell = cleaned["cells"][0]
    assert "execution" not in code_cell["metadata"]
    assert code_cell["metadata"]["tags"] == ["a-garder"]
    assert code_cell["outputs"][0]["text"] == ["é\n"]
    assert cleaned["metadata"]["kernelspec"]["name"] == "python3"
    nbformat.validate(nbformat.reads(json.dumps(cleaned), as_version=4))


def test_clean_merges_split_print_outputs_without_changing_the_text():
    nb = make_notebook()
    cell = nb.cells[0]
    cell.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="Versions :"),
        nbformat.v4.new_output("stream", name="stdout", text=" {'python': '3.11'}\nfin\n"),
        nbformat.v4.new_output("stream", name="stderr", text="avertissement\n"),
    ]
    cleaned = json.loads(nettoyer_notebook.clean(nbformat.writes(nb)))
    outputs = cleaned["cells"][0]["outputs"]
    assert len(outputs) == 2  # les deux morceaux stdout fusionnés, stderr à part
    assert "".join(outputs[0]["text"]) == "Versions : {'python': '3.11'}\nfin\n"
    assert outputs[0]["text"] == ["Versions : {'python': '3.11'}\n", "fin\n"]
    assert outputs[1]["name"] == "stderr"

    # Le même print reçu en un seul morceau donne exactement le même fichier.
    cell.outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="Versions : {'python': '3.11'}\nfin\n"),
        nbformat.v4.new_output("stream", name="stderr", text="avertissement\n"),
    ]
    assert nettoyer_notebook.clean(nbformat.writes(nb)) == nettoyer_notebook.clean(json.dumps(cleaned))


def test_clean_keeps_nbformat_layout_and_is_idempotent():
    nb = make_notebook()
    del nb.cells[0].metadata["execution"]
    text = nbformat.writes(nb) + "\n"
    once = nettoyer_notebook.clean(text)
    assert once == text  # notebook déjà propre : octets identiques
    assert nettoyer_notebook.clean(once) == once
