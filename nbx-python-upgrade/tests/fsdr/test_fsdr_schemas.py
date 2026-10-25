"""Every generated file matches the data contract exactly (names, case, raw types) and its JSON Schema."""
from __future__ import annotations

import json

import jsonschema
import pytest

from nbx.fsdr import config, schemas
from nbx.fsdr.datasource import read_json_records


def test_contract_has_45_tables_and_one_shared_model_schema():
    assert len(schemas.TABLES) == 45
    model = [s for s in schemas.TABLES.values() if s.family == "model_score"]
    assert len(model) == len(config.MODEL_SCORE_REGISTRY) == 11
    assert {tuple(s.fields) for s in model} == {tuple(schemas.MODEL_SCORE_FIELDS)}


def test_production_quirks_are_kept():
    assert "UBSCRIPTION_ID" in schemas.TABLES["base_prepaid"].field_names
    assert "activities_auto_m2" in schemas.TABLES and "activities_m2_auto" not in schemas.TABLES
    assert "Access_Method_id" in schemas.TABLES["location_auto"].field_names
    cat = schemas.TABLES["btl_catalog"]
    assert {"Price data", "Price voice"} <= set(cat.field_names) and set(cat.types().values()) == {"string"}
    assert set(schemas.TABLES[config.TARGET_SAMPLE_TABLE].types().values()) == {"string"}
    assert schemas.TABLE_ALIASES["activities_w1_score_hfds"] == "activities_w1_score"


def test_generated_tables_match_the_contract(small_source):
    assert set(small_source.tables()) == set(schemas.TABLES)
    for name, schema in schemas.TABLES.items():
        df = small_source.read(name)
        assert list(df.columns) == schema.field_names, name
        assert len(df) > 0, name
        for f in schema.fields:
            bad, examples = schemas.type_errors(df[f.name], f.raw_type)
            assert bad == 0, f"{name}.{f.name} ({f.raw_type}): {examples}"
            if f.key:
                assert df[f.name].notna().all(), f"null key {name}.{f.name}"


def test_messy_production_types_are_kept(small_source):
    cat = small_source.read("btl_catalog")
    assert all(isinstance(v, str) for v in cat["price"]) and all(isinstance(v, str) for v in cat["validity"])
    score = small_source.read(config.MODEL_SCORE_REGISTRY[0].table)
    assert all(isinstance(v, int) for v in score["access_method_val"])        # bigint MSISDN in score files
    assert all(isinstance(v, str) for v in small_source.read("base_prepaid")["ACCESS_METHOD_VAL"])
    assert all(isinstance(v, float) for v in small_source.read("subs_amt_auto")["ACCOUNT_ID"])   # double in subs_amt


def test_written_files_validate_against_exported_json_schemas(small_folder, tmp_path):
    schemas.export_json_schemas(tmp_path)
    assert (tmp_path / "_index.json").exists()
    for name in schemas.TABLES:
        schema = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        records = read_json_records(small_folder / f"{name}.json")
        jsonschema.validate(records[:200], schema)


def test_json_schema_rejects_a_string_key():
    schema = schemas.json_schema("base_prepaid")
    rec = {f: None for f in schemas.TABLES["base_prepaid"].field_names}
    rec.update({"ACCESS_METHOD_VAL": "91234567", "ACCESS_METHOD_ID": "1001", "ACCOUNT_ID": 1, "PARTY_ID": 2})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([rec], schema)
