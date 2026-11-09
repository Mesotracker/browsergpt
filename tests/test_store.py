from __future__ import annotations

from chainforge import cgpt
from chainforge.store import ModelStore, safe_name
from tests.helpers import tiny_model, tiny_tokenizer


def test_safe_name_strips_paths_and_extensions():
    assert safe_name("../../evil.cgpt") == "evil"
    assert safe_name("my model.notes") == "my-model.notes"
    assert safe_name("") == "model"


def test_import_honours_requested_name(tmp_path):
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer)
    archive = tmp_path / "original.cgpt"
    cgpt.save(archive, model, tokenizer, name="original")
    store = ModelStore(tmp_path / "models")

    result = store.add_file(archive, "renamed")
    assert result["name"] == "renamed"
    assert (tmp_path / "models" / "renamed.cgpt").is_file()
    assert [item["name"] for item in store.list_models()] == ["renamed"]


def test_duplicate_names_are_suffixed(tmp_path):
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer)
    archive = tmp_path / "source.cgpt"
    cgpt.save(archive, model, tokenizer, name="source")
    store = ModelStore(tmp_path / "models")

    first = store.add_file(archive, "notes")
    second = store.add_file(archive, "notes")
    assert first["name"] == "notes"
    assert second["name"] != "notes"
    assert second["name"].startswith("notes-")
    assert {item["name"] for item in store.list_models()} == {first["name"], second["name"]}
