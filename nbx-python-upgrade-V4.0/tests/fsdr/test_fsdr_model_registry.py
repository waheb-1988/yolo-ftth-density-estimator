"""Adding a model score is a config change only: schema, generator and validation pick it up.
(Phase 3 extends this test to the features, the VW format and the model-compatibility check.)"""
from __future__ import annotations

from datetime import date

import pytest

from nbx.fsdr import config, schemas
from nbx.fsdr.datasource import SyntheticSource
from nbx.fsdr.validation import validate

DUMMY = config.ModelScore("DAIM_DUMMY_TWELFTH_MODEL.prediction", "PROBA_DUMMY", "max", "dummy_signal")


@pytest.fixture
def twelve_models(monkeypatch):
    monkeypatch.setattr(config, "MODEL_SCORE_REGISTRY", config.MODEL_SCORE_REGISTRY + [DUMMY])
    schemas.refresh()
    yield
    monkeypatch.undo()
    schemas.refresh()


def test_registry_derives_eleven_files_and_nine_features():
    assert len(config.MODEL_SCORE_REGISTRY) == 11
    assert config.proba_features() == ["PROBA_CHURN", "PROBA_REACT_30", "PROBA_REACT_15", "PROBA_APP_DATA",
                                       "PROBA_APP_VOICE", "PROBA_CHURN_1D", "PROBA_REV_DECR", "PROBA_FRAG_DATA",
                                       "PROBA_FRAG_VOICE"]


def test_a_twelfth_model_flows_through_config_only(twelve_models):
    assert DUMMY.table in schemas.TABLES and schemas.TABLES[DUMMY.table].family == "model_score"
    assert config.proba_features()[-1] == "PROBA_DUMMY"
    src = SyntheticSource(n=300, run_date=date(2026, 10, 21), seed=12)
    df = src.read(DUMMY.table)
    assert list(df.columns) == schemas.TABLES[DUMMY.table].field_names and len(df) > 150
    rep = validate(src)
    assert rep.status == "ok" and DUMMY.table in rep.coverage


def test_registry_restored_after_the_test():
    assert DUMMY.table not in schemas.TABLES
